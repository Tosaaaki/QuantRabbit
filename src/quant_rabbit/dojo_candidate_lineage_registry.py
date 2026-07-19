"""Append-only local lineage discipline for DOJO bot trainer studies.

The registry limits one candidate lineage to three sequential TRAIN attempts and
fourteen unique candidate configurations/proposals.  Every attempt is bound to
the exact sealed-study bytes; every later attempt also binds the immediately
preceding trainer evaluation and its exact artifact bytes.  The registry rebuilds
the evaluation's contract, risk receipt, visible gates, score, and ranking, but it
does not replay the evaluation's cell ledgers itself.

This is a local first-write discipline mechanism.  The owner of the filesystem
can still delete and recreate the whole directory or recompute a replacement
chain.  Without an external signed monotonic witness and independent custody it
is not proof, promotion evidence, or live-trading authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final

from quant_rabbit.dojo_bot_trainer import (
    EVALUATION_CONTRACT,
    REQUIRED_COST_ARMS,
    REQUIRED_INTRABAR_PATHS,
    DojoBotTrainerError,
    verify_sealed_study,
)
from quant_rabbit.dojo_bot_catalog import bot_config_risk_vector


EVENT_CONTRACT: Final = "QR_DOJO_CANDIDATE_LINEAGE_EVENT_V1"
STATUS_CONTRACT: Final = "QR_DOJO_CANDIDATE_LINEAGE_STATUS_V1"
SCHEMA_VERSION: Final = 1
MAX_ATTEMPTS: Final = 3
MAX_UNIQUE_CONFIGS: Final = 14
MAX_UNIQUE_PROPOSALS: Final = 14
MAX_EVENTS: Final = 1 + (MAX_ATTEMPTS * 2)
MAX_EVENT_BYTES: Final = 512 * 1024
MAX_ARTIFACT_BYTES: Final = 64 * 1024 * 1024
# Evaluation summaries round metrics to 12 decimals and scores to 6.  These
# tolerances are serialization reconciliation units, not market/risk knobs; a
# later evaluation schema should replace them together with its precision.
SUMMARY_METRIC_TOLERANCE: Final = 1e-9
SUMMARY_SCORE_TOLERANCE: Final = 1e-6

_HEX64: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_EVENT_NAME: Final = re.compile(r"[0-9]{6}\.json\Z")
_EVENT_TYPES: Final = frozenset(
    {"GENESIS", "STUDY_SEALED", "RESULT_BOUND", "NEXT_STUDY_SEALED"}
)
_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "registry_id",
        "sequence",
        "event_type",
        "event_at_utc",
        "previous_event_sha256",
        "body",
        "external_witness_status",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "event_sha256",
    }
)
_GENESIS_KEYS: Final = frozenset(
    {
        "created_by",
        "lineage_prefix",
        "monotonicity_scope",
        "max_attempts",
        "max_unique_configs",
        "max_unique_proposals",
    }
)
_STUDY_KEYS: Final = frozenset(
    {
        "attempt_ordinal",
        "study_id",
        "lineage_prefix",
        "window",
        "window_key_sha256",
        "study_sha256",
        "study_artifact_relpath",
        "study_artifact_sha256",
        "study_artifact_size_bytes",
        "candidate_count",
        "candidate_ids",
        "config_sha256s",
        "proposal_sha256s",
        "cumulative_unique_config_sha256s",
        "cumulative_unique_proposal_sha256s",
        "previous_attempt_evaluation_binding",
    }
)
_RESULT_KEYS: Final = frozenset(
    {
        "attempt_ordinal",
        "study_sha256",
        "evaluation_sha256",
        "evaluation_artifact_relpath",
        "evaluation_artifact_sha256",
        "evaluation_artifact_size_bytes",
        "verified_trainer_evaluation",
    }
)
_PREVIOUS_EVALUATION_KEYS: Final = frozenset(
    {
        "attempt_ordinal",
        "evaluation_sha256",
        "evaluation_artifact_sha256",
        "evaluation_artifact_size_bytes",
    }
)
_EVALUATION_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "study_sha256",
        "classification",
        "fixed_denominator",
        "candidate_evaluations",
        "diagnostic_ranking",
        "rank_eligible_candidate_ids",
        "unranked_candidate_ids",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "evaluation_sha256",
    }
)
_DENOMINATOR_KEYS: Final = frozenset(
    {
        "candidate_count",
        "intrabar_paths",
        "cost_arms",
        "expected_cell_count",
        "observed_cell_count",
        "coordinate_receipts_complete",
        "execution_success_complete",
    }
)
_CANDIDATE_EVALUATION_KEYS: Final = frozenset(
    {
        "candidate_id",
        "status",
        "diagnostic_rank_eligible",
        "diagnostic_score",
        "risk_policy_receipt",
        "proposer_risk_claim_ignored",
        "gate_blockers",
        "failed_coordinates",
        "coordinate_worst",
        "cost_retention_by_intrabar",
        "capital_productivity_by_cell",
        "mtm_complete",
        "mtm_incomplete_uses_realized_dd_for_train_diagnostic_only",
        "lopo_replay_complete",
        "promotion_gate_passed",
        "promotion_blockers",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
    }
)
_COORDINATE_WORST_KEYS: Final = frozenset(
    {
        "terminal_net_jpy",
        "realized_max_drawdown_fraction",
        "normal_effective_drawdown_fraction",
        "stress_effective_drawdown_fraction",
        "peak_margin_usage_fraction",
        "margin_reject_rate",
        "cost_retention",
        "pair_positive_share",
        "pair_hhi",
        "effective_positive_pairs",
        "leave_one_pair_out_net_jpy",
        "capital_productivity_per_margin_day",
    }
)
_LIMITATIONS: Final = (
    "LOCAL_OWNER_CAN_DELETE_OR_RECREATE_ENTIRE_LEDGER",
    "LOCAL_OWNER_CAN_RECOMPUTE_A_REPLACEMENT_CHAIN",
    "LOCAL_OWNER_CAN_CREATE_A_PARALLEL_REGISTRY_DIRECTORY",
    "REGISTRY_DOES_NOT_REPLAY_TRAINER_CELL_LEDGERS",
    "EXTERNAL_SIGNED_MONOTONIC_WITNESS_AND_INDEPENDENT_CUSTODY_ABSENT",
)


class CandidateLineageError(ValueError):
    """The lineage registry, transition, or artifact binding is invalid."""


@dataclass(frozen=True)
class CandidateLineageSnapshot:
    """Verified view of the local lineage registry and its bound artifacts."""

    registry_id: str
    lineage_prefix: str
    event_count: int
    latest_sequence: int
    latest_event_sha256: str
    latest_event_at_utc: str
    studies: tuple[Mapping[str, Any], ...]
    results: tuple[Mapping[str, Any], ...]
    cumulative_unique_config_sha256s: tuple[str, ...]
    cumulative_unique_proposal_sha256s: tuple[str, ...]
    events: tuple[Mapping[str, Any], ...]
    external_witness_status: str = "ABSENT"
    proof_eligible: bool = False
    promotion_eligible: bool = False
    live_permission: bool = False
    order_authority: str = "NONE"
    broker_mutation_allowed: bool = False


def initialize_registry(
    events_dir: Path,
    *,
    artifact_root: Path,
    registry_id: str,
    lineage_prefix: str,
    created_by: str,
    event_at_utc: datetime | str,
) -> CandidateLineageSnapshot:
    """Create one genesis event in a new or empty real directory."""

    _open_artifact_root(Path(artifact_root), close=True)
    body = {
        "created_by": _text(created_by, "created_by"),
        "lineage_prefix": _prefix(lineage_prefix),
        "monotonicity_scope": "LOCAL_FIRST_WRITE_ONLY_NO_EXTERNAL_WITNESS",
        "max_attempts": MAX_ATTEMPTS,
        "max_unique_configs": MAX_UNIQUE_CONFIGS,
        "max_unique_proposals": MAX_UNIQUE_PROPOSALS,
    }
    directory_fd = _open_events_directory(Path(events_dir), create=True)
    try:
        if _event_names(directory_fd):
            raise CandidateLineageError("candidate lineage is already initialized")
        event = _new_event(
            registry_id=_identifier(registry_id, "registry_id"),
            sequence=0,
            event_type="GENESIS",
            event_at_utc=event_at_utc,
            previous_event_sha256=None,
            body=body,
        )
        _write_event_exclusive(directory_fd, "000000.json", event)
    finally:
        os.close(directory_fd)
    return verify_registry(Path(events_dir), artifact_root=Path(artifact_root))


def seal_study_attempt(
    events_dir: Path,
    *,
    artifact_root: Path,
    sealed_study_path: Path,
    expected_tip_sha256: str,
    event_at_utc: datetime | str,
    previous_evaluation_sha256: str | None = None,
    previous_evaluation_artifact_sha256: str | None = None,
    previous_evaluation_artifact_size_bytes: int | None = None,
) -> CandidateLineageSnapshot:
    """Bind the next sealed study, enforcing ordinal and prior-result lineage."""

    snapshot = verify_registry(Path(events_dir), artifact_root=Path(artifact_root))
    _expected_tip(snapshot, expected_tip_sha256)
    if len(snapshot.studies) >= MAX_ATTEMPTS:
        raise CandidateLineageError("candidate lineage attempt limit is exhausted")
    if len(snapshot.studies) != len(snapshot.results):
        raise CandidateLineageError(
            "the current attempt must bind its result before another study"
        )

    previous_binding: dict[str, Any] | None = None
    if snapshot.results:
        result = snapshot.results[-1]
        supplied = {
            "attempt_ordinal": result["attempt_ordinal"],
            "evaluation_sha256": _required_sha(
                previous_evaluation_sha256, "previous_evaluation_sha256"
            ),
            "evaluation_artifact_sha256": _required_sha(
                previous_evaluation_artifact_sha256,
                "previous_evaluation_artifact_sha256",
            ),
            "evaluation_artifact_size_bytes": _positive_integer(
                previous_evaluation_artifact_size_bytes,
                "previous_evaluation_artifact_size_bytes",
            ),
        }
        expected = _previous_binding(result)
        if supplied != expected:
            raise CandidateLineageError(
                "next attempt does not bind the immediately preceding verified "
                "trainer evaluation bytes"
            )
        previous_binding = supplied
    elif any(
        value is not None
        for value in (
            previous_evaluation_sha256,
            previous_evaluation_artifact_sha256,
            previous_evaluation_artifact_size_bytes,
        )
    ):
        raise CandidateLineageError("first attempt cannot claim a previous evaluation")

    reference = _read_artifact_path(Path(artifact_root), Path(sealed_study_path))
    body = _derive_study_body(
        reference=reference,
        lineage_prefix=snapshot.lineage_prefix,
        previous_binding=previous_binding,
        prior_studies=snapshot.studies,
    )
    expected_ordinal = len(snapshot.studies) + 1
    if body["attempt_ordinal"] != expected_ordinal:
        raise CandidateLineageError(
            "study attempt ordinal is stale, duplicated, or contains a gap"
        )
    event_type = "STUDY_SEALED" if expected_ordinal == 1 else "NEXT_STUDY_SEALED"
    return _append(
        Path(events_dir),
        Path(artifact_root),
        snapshot,
        event_type,
        body,
        event_at_utc,
    )


def bind_result(
    events_dir: Path,
    *,
    artifact_root: Path,
    evaluation_path: Path,
    expected_tip_sha256: str,
    event_at_utc: datetime | str,
) -> CandidateLineageSnapshot:
    """Bind one verified trainer evaluation to the pending sealed study."""

    snapshot = verify_registry(Path(events_dir), artifact_root=Path(artifact_root))
    _expected_tip(snapshot, expected_tip_sha256)
    if not snapshot.studies or len(snapshot.studies) != len(snapshot.results) + 1:
        raise CandidateLineageError("there is no single pending study result to bind")
    study = snapshot.studies[-1]
    reference = _read_artifact_path(Path(artifact_root), Path(evaluation_path))
    sealed_study_reference = _read_bound_artifact(
        Path(artifact_root),
        study["study_artifact_relpath"],
        expected_sha256=study["study_artifact_sha256"],
        expected_size=study["study_artifact_size_bytes"],
    )
    evaluation = _verify_evaluation(
        reference.value,
        study,
        sealed_study_artifact=sealed_study_reference.value,
    )
    body = {
        "attempt_ordinal": study["attempt_ordinal"],
        "study_sha256": study["study_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "evaluation_artifact_relpath": reference.relpath,
        "evaluation_artifact_sha256": reference.sha256,
        "evaluation_artifact_size_bytes": reference.size_bytes,
        "verified_trainer_evaluation": True,
    }
    return _append(
        Path(events_dir),
        Path(artifact_root),
        snapshot,
        "RESULT_BOUND",
        body,
        event_at_utc,
    )


def verify_registry(
    events_dir: Path, *, artifact_root: Path
) -> CandidateLineageSnapshot:
    """Verify exact slots, SHA chain, state grammar, budgets, and artifacts."""

    root = Path(artifact_root)
    _open_artifact_root(root, close=True)
    directory_fd = _open_events_directory(Path(events_dir), create=False)
    try:
        names = _event_names(directory_fd)
        if not names:
            raise CandidateLineageError("candidate lineage lacks genesis")
        if len(names) > MAX_EVENTS:
            raise CandidateLineageError("candidate lineage event limit exceeded")
        expected_names = [f"{sequence:06d}.json" for sequence in range(len(names))]
        if names != expected_names:
            raise CandidateLineageError("candidate lineage has an ordinal gap or fork")
        events: list[dict[str, Any]] = []
        total_bytes = 0
        for name in names:
            event, size = _read_event(directory_fd, name)
            events.append(event)
            total_bytes += size
        if total_bytes > MAX_EVENTS * MAX_EVENT_BYTES:
            raise CandidateLineageError("candidate lineage byte limit exceeded")
        if _event_names(directory_fd) != names:
            raise CandidateLineageError("candidate lineage changed while being read")
    finally:
        os.close(directory_fd)
    return _replay(events, artifact_root=root)


def status_artifact(events_dir: Path, *, artifact_root: Path) -> dict[str, Any]:
    """Return a fail-closed diagnostic status with all authorities disabled."""

    snapshot = verify_registry(Path(events_dir), artifact_root=Path(artifact_root))
    if len(snapshot.studies) == len(snapshot.results):
        state = (
            "COMPLETE" if len(snapshot.studies) == MAX_ATTEMPTS else "READY_FOR_STUDY"
        )
    else:
        state = "AWAITING_RESULT"
    return {
        "contract": STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": snapshot.registry_id,
        "lineage_prefix": snapshot.lineage_prefix,
        "state": state,
        "attempt_count": len(snapshot.studies),
        "result_count": len(snapshot.results),
        "max_attempts": MAX_ATTEMPTS,
        "unique_config_count": len(snapshot.cumulative_unique_config_sha256s),
        "unique_proposal_count": len(snapshot.cumulative_unique_proposal_sha256s),
        "max_unique_configs": MAX_UNIQUE_CONFIGS,
        "max_unique_proposals": MAX_UNIQUE_PROPOSALS,
        "latest_sequence": snapshot.latest_sequence,
        "latest_event_sha256": snapshot.latest_event_sha256,
        "external_witness_status": "ABSENT",
        "limitations": list(_LIMITATIONS),
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }


@dataclass(frozen=True)
class _ArtifactReference:
    relpath: str
    sha256: str
    size_bytes: int
    value: Mapping[str, Any]


def _derive_study_body(
    *,
    reference: _ArtifactReference,
    lineage_prefix: str,
    previous_binding: Mapping[str, Any] | None,
    prior_studies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        source_digests = reference.value.get("source_digests")
        if not isinstance(source_digests, Mapping):
            raise CandidateLineageError("sealed study lacks source digests")
        sealed = verify_sealed_study(reference.value, source_digests)
    except DojoBotTrainerError as exc:
        raise CandidateLineageError(f"sealed study verification failed: {exc}") from exc
    study = sealed["study"]
    budget = study["search_budget"]
    if budget["total_attempts_in_lineage"] != MAX_ATTEMPTS:
        raise CandidateLineageError(
            f"study must pre-register exactly {MAX_ATTEMPTS} total attempts"
        )
    candidates = study["candidates"]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if any(
        not candidate_id.startswith(lineage_prefix) for candidate_id in candidate_ids
    ):
        raise CandidateLineageError(
            "candidate id is outside the registered lineage prefix"
        )
    config_sha256s = sorted(candidate["config_sha256"] for candidate in candidates)
    proposal_sha256s = sorted(candidate["proposal_sha256"] for candidate in candidates)
    if len(set(config_sha256s)) != len(config_sha256s):
        raise CandidateLineageError("study repeats a config in the same lineage/window")
    if len(set(proposal_sha256s)) != len(proposal_sha256s):
        raise CandidateLineageError(
            "study repeats a proposal in the same lineage/window"
        )

    window = study["window"]
    # A market window is the economic opportunity interval.  Mutable labels
    # such as corpus_id (or a re-exported corpus digest) must not mint another
    # chance to retry the same configuration on the same timestamps.
    window_key = _canonical_sha(
        {
            "lineage_prefix": lineage_prefix,
            "start_utc": window["start_utc"],
            "end_utc": window["end_utc"],
        }
    )
    for prior in prior_studies:
        if prior["window_key_sha256"] != window_key:
            continue
        if set(config_sha256s).intersection(prior["config_sha256s"]):
            raise CandidateLineageError(
                "duplicate config is forbidden for the same lineage prefix/window"
            )
        if set(proposal_sha256s).intersection(prior["proposal_sha256s"]):
            raise CandidateLineageError(
                "duplicate proposal is forbidden for the same lineage prefix/window"
            )

    prior_configs = {sha for prior in prior_studies for sha in prior["config_sha256s"]}
    prior_proposals = {
        sha for prior in prior_studies for sha in prior["proposal_sha256s"]
    }
    cumulative_configs = sorted(prior_configs.union(config_sha256s))
    cumulative_proposals = sorted(prior_proposals.union(proposal_sha256s))
    if len(cumulative_configs) > MAX_UNIQUE_CONFIGS:
        raise CandidateLineageError("cumulative unique config budget exceeds 14")
    if len(cumulative_proposals) > MAX_UNIQUE_PROPOSALS:
        raise CandidateLineageError("cumulative unique proposal budget exceeds 14")

    normalized_previous = None
    if previous_binding is not None:
        normalized_previous = _normalize_previous_binding(previous_binding)
    return {
        "attempt_ordinal": budget["attempt_ordinal"],
        "study_id": study["study_id"],
        "lineage_prefix": lineage_prefix,
        "window": window,
        "window_key_sha256": window_key,
        "study_sha256": sealed["study_sha256"],
        "study_artifact_relpath": reference.relpath,
        "study_artifact_sha256": reference.sha256,
        "study_artifact_size_bytes": reference.size_bytes,
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "config_sha256s": config_sha256s,
        "proposal_sha256s": proposal_sha256s,
        "cumulative_unique_config_sha256s": cumulative_configs,
        "cumulative_unique_proposal_sha256s": cumulative_proposals,
        "previous_attempt_evaluation_binding": normalized_previous,
    }


def _verify_evaluation(
    value: Mapping[str, Any],
    study: Mapping[str, Any],
    *,
    sealed_study_artifact: Mapping[str, Any],
) -> Mapping[str, Any]:
    row = _exact_mapping(value, _EVALUATION_KEYS, "trainer evaluation")
    if (
        row["contract"] != EVALUATION_CONTRACT
        or isinstance(row["schema_version"], bool)
        or row["schema_version"] != 1
    ):
        raise CandidateLineageError("trainer evaluation contract/version drifted")
    if row["study_sha256"] != study["study_sha256"]:
        raise CandidateLineageError("trainer evaluation belongs to another study")
    source_digests = sealed_study_artifact.get("source_digests")
    if not isinstance(source_digests, Mapping):
        raise CandidateLineageError("bound sealed study lacks source digests")
    try:
        sealed_study = verify_sealed_study(sealed_study_artifact, source_digests)
    except DojoBotTrainerError as exc:
        raise CandidateLineageError(
            f"bound sealed study verification failed: {exc}"
        ) from exc
    if sealed_study["study_sha256"] != study["study_sha256"]:
        raise CandidateLineageError("bound sealed study digest drifted")
    candidates_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in sealed_study["study"]["candidates"]
    }
    if (
        row["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY"
        or row["proof_eligible"] is not False
        or row["promotion_eligible"] is not False
        or row["live_permission"] is not False
        or row["order_authority"] != "NONE"
        or row["broker_mutation_allowed"] is not False
    ):
        raise CandidateLineageError("trainer evaluation safety boundary drifted")

    denominator = _exact_mapping(
        row["fixed_denominator"], _DENOMINATOR_KEYS, "fixed denominator"
    )
    candidate_count = study["candidate_count"]
    expected_cells = (
        candidate_count * len(REQUIRED_INTRABAR_PATHS) * len(REQUIRED_COST_ARMS)
    )
    if (
        denominator["candidate_count"] != candidate_count
        or denominator["intrabar_paths"] != list(REQUIRED_INTRABAR_PATHS)
        or denominator["cost_arms"] != list(REQUIRED_COST_ARMS)
        or denominator["expected_cell_count"] != expected_cells
        or isinstance(denominator["candidate_count"], bool)
        or denominator["observed_cell_count"] != expected_cells
        or isinstance(denominator["expected_cell_count"], bool)
        or isinstance(denominator["observed_cell_count"], bool)
        or denominator["coordinate_receipts_complete"] is not True
        or not isinstance(denominator["execution_success_complete"], bool)
    ):
        raise CandidateLineageError("trainer evaluation fixed denominator is invalid")

    evaluations = row["candidate_evaluations"]
    if not isinstance(evaluations, list):
        raise CandidateLineageError("candidate evaluations must be a list")
    evaluation_ids: list[str] = []
    eligible_from_rows: list[str] = []
    unranked_from_rows: list[str] = []
    failed_coordinate_seen = False
    for evaluation in evaluations:
        mapping = _exact_mapping(
            evaluation, _CANDIDATE_EVALUATION_KEYS, "candidate evaluation"
        )
        candidate_id = _identifier(mapping["candidate_id"], "candidate_id")
        evaluation_ids.append(candidate_id)
        eligible = mapping["diagnostic_rank_eligible"]
        if not isinstance(eligible, bool):
            raise CandidateLineageError("diagnostic rank eligibility must be boolean")
        expected_status = "TRAIN_DIAGNOSTIC_PASS" if eligible else "TRAIN_REJECT"
        if mapping["status"] != expected_status:
            raise CandidateLineageError("candidate evaluation status is inconsistent")
        gate_blockers = _unique_text_list(
            mapping["gate_blockers"], "candidate evaluation gate_blockers"
        )
        failed_coordinates = _unique_text_list(
            mapping["failed_coordinates"],
            "candidate evaluation failed_coordinates",
        )
        promotion_blockers = _unique_text_list(
            mapping["promotion_blockers"],
            "candidate evaluation promotion_blockers",
        )
        if eligible != (not gate_blockers):
            raise CandidateLineageError(
                "candidate eligibility must equal an empty blocker set"
            )
        if eligible:
            score = _finite_number(
                mapping["diagnostic_score"], "candidate diagnostic score"
            )
            if score < 0.0 or score > 100.0:
                raise CandidateLineageError(
                    "candidate diagnostic score is out of range"
                )
            eligible_from_rows.append(candidate_id)
        else:
            if mapping["diagnostic_score"] is not None:
                raise CandidateLineageError(
                    "rejected candidate carries diagnostic score"
                )
            unranked_from_rows.append(candidate_id)
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            raise CandidateLineageError("candidate evaluation is outside sealed study")
        expected_risk = bot_config_risk_vector(
            candidate["config"],
            stress_slippage_pips_per_fill=sealed_study["study"]["cost_arms"]["STRESS"][
                "slippage_pips_per_fill"
            ],
        )
        if mapping["risk_policy_receipt"] != expected_risk:
            raise CandidateLineageError(
                "candidate risk policy receipt is not trainer-derived"
            )
        if (
            gate_blockers[: len(expected_risk["blocker_codes"])]
            != expected_risk["blocker_codes"]
        ):
            raise CandidateLineageError(
                "candidate blockers omit or reorder risk-policy blockers"
            )
        coordinate_worst = _verify_coordinate_summary(mapping)
        for coordinate in failed_coordinates:
            parts = coordinate.split(":", 2)
            if (
                len(parts) != 3
                or parts[0] not in REQUIRED_INTRABAR_PATHS
                or parts[1] not in REQUIRED_COST_ARMS
                or not parts[2]
            ):
                raise CandidateLineageError(
                    "candidate failed coordinate is outside the fixed denominator"
                )
        if bool(failed_coordinates) != ("RUNNER_CELL_FAILURE" in gate_blockers):
            raise CandidateLineageError(
                "runner failure blocker and failed coordinates diverged"
            )
        failed_coordinate_seen = failed_coordinate_seen or bool(failed_coordinates)
        expected_promotion_blockers = [
            "WORN_HISTORICAL_TRAIN_ONLY",
            "PROSPECTIVE_FORWARD_EVIDENCE_REQUIRED",
        ]
        if mapping["mtm_complete"] is False:
            expected_promotion_blockers.append("CONTINUOUS_MTM_EVIDENCE_INCOMPLETE")
        if mapping["lopo_replay_complete"] is False:
            expected_promotion_blockers.append("COUNTERFACTUAL_LOPO_INCOMPLETE")
        if promotion_blockers != expected_promotion_blockers:
            raise CandidateLineageError(
                "candidate promotion blockers are not trainer-derived"
            )
        if (mapping["mtm_complete"] is False) != (
            "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE" in gate_blockers
        ):
            raise CandidateLineageError(
                "MTM completeness and candidate blocker diverged"
            )
        if (mapping["lopo_replay_complete"] is False) != (
            "COUNTERFACTUAL_LOPO_INCOMPLETE" in gate_blockers
        ):
            raise CandidateLineageError(
                "LOPO completeness and candidate blocker diverged"
            )
        _verify_summary_gate_consistency(
            mapping,
            coordinate_worst=coordinate_worst,
            gate_blockers=gate_blockers,
            sealed_study=sealed_study,
        )
        if (
            mapping["proposer_risk_claim_ignored"] is not True
            or not isinstance(mapping["mtm_complete"], bool)
            or mapping["mtm_incomplete_uses_realized_dd_for_train_diagnostic_only"]
            is not False
            or not isinstance(mapping["lopo_replay_complete"], bool)
            or mapping["promotion_gate_passed"] is not False
            or mapping["proof_eligible"] is not False
            or mapping["promotion_eligible"] is not False
            or mapping["live_permission"] is not False
            or mapping["order_authority"] != "NONE"
        ):
            raise CandidateLineageError(
                "candidate evaluation safety boundary is inconsistent"
            )
    expected_ids = list(study["candidate_ids"])
    if evaluation_ids != expected_ids:
        raise CandidateLineageError(
            "trainer evaluation candidate denominator is incomplete or reordered"
        )
    ranked = _identifier_list(row["rank_eligible_candidate_ids"], "rank eligible ids")
    unranked = _identifier_list(row["unranked_candidate_ids"], "unranked ids")
    diagnostic = _identifier_list(row["diagnostic_ranking"], "diagnostic ranking")
    expected_ranking = sorted(
        eligible_from_rows,
        key=lambda candidate_id: (
            -float(
                next(
                    evaluation["diagnostic_score"]
                    for evaluation in evaluations
                    if evaluation["candidate_id"] == candidate_id
                )
            ),
            candidate_id,
        ),
    )
    if diagnostic != ranked or ranked != expected_ranking:
        raise CandidateLineageError("diagnostic ranking and eligible ids diverged")
    if set(ranked).intersection(unranked) or sorted(ranked + unranked) != expected_ids:
        raise CandidateLineageError("ranked/unranked candidates do not partition study")
    if set(ranked) != set(eligible_from_rows) or unranked != unranked_from_rows:
        raise CandidateLineageError("candidate row eligibility diverged from ranking")
    if denominator["execution_success_complete"] is failed_coordinate_seen:
        raise CandidateLineageError(
            "execution success flag and failed coordinates diverged"
        )

    claimed = _required_sha(row["evaluation_sha256"], "evaluation_sha256")
    body = {key: item for key, item in row.items() if key != "evaluation_sha256"}
    if _canonical_sha(body) != claimed:
        raise CandidateLineageError("trainer evaluation digest mismatch")
    return row


def _replay(
    events: Sequence[Mapping[str, Any]], *, artifact_root: Path
) -> CandidateLineageSnapshot:
    registry_id: str | None = None
    lineage_prefix: str | None = None
    previous_sha: str | None = None
    previous_time: datetime | None = None
    studies: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    verified_events: list[dict[str, Any]] = []

    for sequence, raw in enumerate(events):
        event = _exact_mapping(raw, _TOP_LEVEL_KEYS, "candidate lineage event")
        if (
            event["contract"] != EVENT_CONTRACT
            or isinstance(event["schema_version"], bool)
            or event["schema_version"] != SCHEMA_VERSION
        ):
            raise CandidateLineageError("event contract/version drifted")
        if isinstance(event["sequence"], bool) or event["sequence"] != sequence:
            raise CandidateLineageError("event sequence drifted")
        if event["previous_event_sha256"] != previous_sha:
            raise CandidateLineageError("event SHA chain drifted")
        event_sha = _required_sha(event["event_sha256"], "event_sha256")
        body_without_sha = {
            key: item for key, item in event.items() if key != "event_sha256"
        }
        if _canonical_sha(body_without_sha) != event_sha:
            raise CandidateLineageError("event_sha256 mismatch")
        if (
            event["external_witness_status"] != "ABSENT"
            or event["proof_eligible"] is not False
            or event["promotion_eligible"] is not False
            or event["live_permission"] is not False
            or event["order_authority"] != "NONE"
            or event["broker_mutation_allowed"] is not False
        ):
            raise CandidateLineageError("event exceeds local diagnostic authority")
        event_type = event["event_type"]
        if event_type not in _EVENT_TYPES:
            raise CandidateLineageError("unsupported candidate lineage event type")
        event_time_text = _utc_text(event["event_at_utc"], "event_at_utc")
        event_time = _parse_utc(event_time_text, "event_at_utc")
        if previous_time is not None and event_time < previous_time:
            raise CandidateLineageError("event time moved backward")

        current_registry = _identifier(event["registry_id"], "registry_id")
        if sequence == 0:
            if event_type != "GENESIS" or previous_sha is not None:
                raise CandidateLineageError("first event must be genesis")
            registry_id = current_registry
            genesis = _normalize_genesis(event["body"])
            lineage_prefix = genesis["lineage_prefix"]
        else:
            if current_registry != registry_id or event_type == "GENESIS":
                raise CandidateLineageError("registry identity or genesis drifted")
            assert lineage_prefix is not None
            if len(studies) == len(results):
                expected_type = "STUDY_SEALED" if not studies else "NEXT_STUDY_SEALED"
                if event_type != expected_type:
                    raise CandidateLineageError("study/result state transition drifted")
                if len(studies) >= MAX_ATTEMPTS:
                    raise CandidateLineageError(
                        "event follows exhausted attempt budget"
                    )
                body = _exact_mapping(event["body"], _STUDY_KEYS, "study binding")
                reference = _read_bound_artifact(
                    artifact_root,
                    body["study_artifact_relpath"],
                    expected_sha256=body["study_artifact_sha256"],
                    expected_size=body["study_artifact_size_bytes"],
                )
                previous_binding = _previous_binding(results[-1]) if results else None
                expected_body = _derive_study_body(
                    reference=reference,
                    lineage_prefix=lineage_prefix,
                    previous_binding=previous_binding,
                    prior_studies=studies,
                )
                if body != expected_body:
                    raise CandidateLineageError("study event binding is not canonical")
                if body["attempt_ordinal"] != len(studies) + 1:
                    raise CandidateLineageError("attempt ordinal is stale or has a gap")
                studies.append(body)
            else:
                if event_type != "RESULT_BOUND":
                    raise CandidateLineageError(
                        "pending study requires exactly one result"
                    )
                body = _exact_mapping(event["body"], _RESULT_KEYS, "result binding")
                study = studies[-1]
                if (
                    body["attempt_ordinal"] != study["attempt_ordinal"]
                    or body["study_sha256"] != study["study_sha256"]
                    or body["verified_trainer_evaluation"] is not True
                ):
                    raise CandidateLineageError(
                        "result does not bind the pending attempt"
                    )
                reference = _read_bound_artifact(
                    artifact_root,
                    body["evaluation_artifact_relpath"],
                    expected_sha256=body["evaluation_artifact_sha256"],
                    expected_size=body["evaluation_artifact_size_bytes"],
                )
                sealed_study_reference = _read_bound_artifact(
                    artifact_root,
                    study["study_artifact_relpath"],
                    expected_sha256=study["study_artifact_sha256"],
                    expected_size=study["study_artifact_size_bytes"],
                )
                evaluation = _verify_evaluation(
                    reference.value,
                    study,
                    sealed_study_artifact=sealed_study_reference.value,
                )
                if evaluation["evaluation_sha256"] != body["evaluation_sha256"]:
                    raise CandidateLineageError(
                        "result evaluation digest binding drifted"
                    )
                results.append(body)

        previous_sha = event_sha
        previous_time = event_time
        verified_events.append(event)

    if (
        registry_id is None
        or lineage_prefix is None
        or previous_sha is None
        or previous_time is None
    ):
        raise CandidateLineageError("candidate lineage lacks verified genesis")
    cumulative_configs = tuple(
        studies[-1]["cumulative_unique_config_sha256s"] if studies else ()
    )
    cumulative_proposals = tuple(
        studies[-1]["cumulative_unique_proposal_sha256s"] if studies else ()
    )
    return CandidateLineageSnapshot(
        registry_id=registry_id,
        lineage_prefix=lineage_prefix,
        event_count=len(verified_events),
        latest_sequence=len(verified_events) - 1,
        latest_event_sha256=previous_sha,
        latest_event_at_utc=_iso(previous_time),
        studies=tuple(studies),
        results=tuple(results),
        cumulative_unique_config_sha256s=cumulative_configs,
        cumulative_unique_proposal_sha256s=cumulative_proposals,
        events=tuple(verified_events),
    )


def _append(
    events_dir: Path,
    artifact_root: Path,
    snapshot: CandidateLineageSnapshot,
    event_type: str,
    body: Mapping[str, Any],
    event_at_utc: datetime | str,
) -> CandidateLineageSnapshot:
    event_time = _utc_text(event_at_utc, "event_at_utc")
    if _parse_utc(event_time, "event_at_utc") < _parse_utc(
        snapshot.latest_event_at_utc, "latest_event_at_utc"
    ):
        raise CandidateLineageError("event time moved backward")
    sequence = snapshot.latest_sequence + 1
    if sequence >= MAX_EVENTS:
        raise CandidateLineageError("candidate lineage event limit exceeded")
    event = _new_event(
        registry_id=snapshot.registry_id,
        sequence=sequence,
        event_type=event_type,
        event_at_utc=event_time,
        previous_event_sha256=snapshot.latest_event_sha256,
        body=body,
    )
    directory_fd = _open_events_directory(events_dir, create=False)
    try:
        _write_event_exclusive(directory_fd, f"{sequence:06d}.json", event)
    finally:
        os.close(directory_fd)
    return verify_registry(events_dir, artifact_root=artifact_root)


def _new_event(
    *,
    registry_id: str,
    sequence: int,
    event_type: str,
    event_at_utc: datetime | str,
    previous_event_sha256: str | None,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    event = {
        "contract": EVENT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": registry_id,
        "sequence": sequence,
        "event_type": event_type,
        "event_at_utc": _utc_text(event_at_utc, "event_at_utc"),
        "previous_event_sha256": previous_event_sha256,
        "body": _snapshot(body),
        "external_witness_status": "ABSENT",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**event, "event_sha256": _canonical_sha(event)}


def _normalize_genesis(value: Any) -> dict[str, Any]:
    body = _exact_mapping(value, _GENESIS_KEYS, "genesis")
    normalized = {
        "created_by": _text(body["created_by"], "created_by"),
        "lineage_prefix": _prefix(body["lineage_prefix"]),
        "monotonicity_scope": body["monotonicity_scope"],
        "max_attempts": body["max_attempts"],
        "max_unique_configs": body["max_unique_configs"],
        "max_unique_proposals": body["max_unique_proposals"],
    }
    expected = {
        "created_by": normalized["created_by"],
        "lineage_prefix": normalized["lineage_prefix"],
        "monotonicity_scope": "LOCAL_FIRST_WRITE_ONLY_NO_EXTERNAL_WITNESS",
        "max_attempts": MAX_ATTEMPTS,
        "max_unique_configs": MAX_UNIQUE_CONFIGS,
        "max_unique_proposals": MAX_UNIQUE_PROPOSALS,
    }
    if normalized != expected or body != normalized:
        raise CandidateLineageError("genesis policy drifted")
    return normalized


def _normalize_previous_binding(value: Any) -> dict[str, Any]:
    body = _exact_mapping(
        value, _PREVIOUS_EVALUATION_KEYS, "previous evaluation binding"
    )
    return {
        "attempt_ordinal": _positive_integer(
            body["attempt_ordinal"], "previous attempt ordinal"
        ),
        "evaluation_sha256": _required_sha(
            body["evaluation_sha256"], "previous evaluation SHA"
        ),
        "evaluation_artifact_sha256": _required_sha(
            body["evaluation_artifact_sha256"], "previous evaluation artifact SHA"
        ),
        "evaluation_artifact_size_bytes": _positive_integer(
            body["evaluation_artifact_size_bytes"],
            "previous evaluation artifact size",
        ),
    }


def _previous_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attempt_ordinal": result["attempt_ordinal"],
        "evaluation_sha256": result["evaluation_sha256"],
        "evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": result["evaluation_artifact_size_bytes"],
    }


def _expected_tip(snapshot: CandidateLineageSnapshot, value: Any) -> None:
    supplied = _required_sha(value, "expected_tip_sha256")
    if supplied != snapshot.latest_event_sha256:
        raise CandidateLineageError("stale lineage tip; reload before appending")


def _read_artifact_path(root: Path, path: Path) -> _ArtifactReference:
    root_absolute = root.absolute()
    candidate = path if path.is_absolute() else root_absolute / path
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise CandidateLineageError("artifact path escapes artifact root") from exc
    relpath = PurePosixPath(*relative.parts).as_posix()
    return _read_bound_artifact(root, relpath)


def _read_bound_artifact(
    root: Path,
    relpath: Any,
    *,
    expected_sha256: Any | None = None,
    expected_size: Any | None = None,
) -> _ArtifactReference:
    path = _artifact_relpath(relpath)
    parts = PurePosixPath(path).parts
    directory_fd = _open_artifact_root(root)
    try:
        for component in parts[:-1]:
            next_fd = _open_directory_component(directory_fd, component)
            os.close(directory_fd)
            directory_fd = next_fd
        raw = _read_regular_file(directory_fd, parts[-1])
    finally:
        os.close(directory_fd)
    sha256 = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    if (
        expected_sha256 is not None
        and _required_sha(expected_sha256, "artifact SHA") != sha256
    ):
        raise CandidateLineageError("bound artifact bytes were rewritten")
    if (
        expected_size is not None
        and _positive_integer(expected_size, "artifact size") != size
    ):
        raise CandidateLineageError("bound artifact size drifted")
    value = _strict_artifact_json(raw)
    return _ArtifactReference(path, sha256, size, value)


def _open_artifact_root(path: Path, *, close: bool = False) -> int:
    candidate = path.absolute()
    try:
        state = candidate.lstat()
    except OSError as exc:
        raise CandidateLineageError(f"cannot stat artifact root: {exc}") from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise CandidateLineageError("artifact root must be a real directory")
    descriptor = _open_real_directory(candidate, "artifact root")
    if close:
        os.close(descriptor)
    return descriptor


def _open_directory_component(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise CandidateLineageError(
            f"artifact path component is not a real directory: {name}: {exc}"
        ) from exc


def _read_regular_file(directory_fd: int, name: str) -> bytes:
    try:
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise CandidateLineageError(f"cannot stat artifact: {exc}") from exc
    if not stat.S_ISREG(state.st_mode) or state.st_size <= 0:
        raise CandidateLineageError("artifact must be a nonempty regular file")
    if state.st_size > MAX_ARTIFACT_BYTES:
        raise CandidateLineageError("artifact byte limit exceeded")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise CandidateLineageError(f"cannot read artifact: {exc}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != state.st_dev
        or before.st_ino != state.st_ino
        or len(raw) != before.st_size
    ):
        raise CandidateLineageError("artifact changed while being read")
    return raw


def _verify_coordinate_summary(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject truncated or non-finite trainer summary surfaces."""

    worst = _exact_mapping(
        mapping["coordinate_worst"],
        _COORDINATE_WORST_KEYS,
        "candidate coordinate_worst",
    )
    required_numbers = (
        "terminal_net_jpy",
        "realized_max_drawdown_fraction",
        "normal_effective_drawdown_fraction",
        "stress_effective_drawdown_fraction",
        "peak_margin_usage_fraction",
        "margin_reject_rate",
        "pair_positive_share",
        "pair_hhi",
        "effective_positive_pairs",
        "leave_one_pair_out_net_jpy",
    )
    for field in required_numbers:
        _finite_number(worst[field], f"candidate coordinate_worst.{field}")
    for field in ("cost_retention", "capital_productivity_per_margin_day"):
        if worst[field] is not None:
            _finite_number(worst[field], f"candidate coordinate_worst.{field}")

    retention = _mapping(
        mapping["cost_retention_by_intrabar"],
        "candidate cost_retention_by_intrabar",
    )
    if set(retention) != set(REQUIRED_INTRABAR_PATHS):
        raise CandidateLineageError(
            "candidate cost retention does not cover every intrabar path"
        )
    for path, value in retention.items():
        if value is not None:
            _finite_number(value, f"candidate cost retention {path}")

    productivity = _mapping(
        mapping["capital_productivity_by_cell"],
        "candidate capital_productivity_by_cell",
    )
    expected_coordinates = {
        f"{path}:{arm}"
        for path in REQUIRED_INTRABAR_PATHS
        for arm in REQUIRED_COST_ARMS
    }
    if set(productivity) != expected_coordinates:
        raise CandidateLineageError(
            "candidate capital productivity does not cover every fixed cell"
        )
    for coordinate, value in productivity.items():
        if value is not None:
            _finite_number(value, f"candidate capital productivity {coordinate}")
    return worst


def _verify_summary_gate_consistency(
    mapping: Mapping[str, Any],
    *,
    coordinate_worst: Mapping[str, Any],
    gate_blockers: Sequence[str],
    sealed_study: Mapping[str, Any],
) -> None:
    """Rebuild every gate and score recoverable from the evaluation summary.

    The weights and rounding mirror the version-1 trainer evaluation contract.
    A trainer scoring change must version the evaluation contract and this
    verifier together rather than silently accepting caller-authored scores.
    """

    study = sealed_study["study"]
    threshold = study["thresholds"]
    retention_values = list(mapping["cost_retention_by_intrabar"].values())
    productivity_values = list(mapping["capital_productivity_by_cell"].values())
    minimum_retention = (
        min(float(value) for value in retention_values)
        if all(value is not None for value in retention_values)
        else None
    )
    minimum_productivity = (
        min(float(value) for value in productivity_values)
        if all(value is not None for value in productivity_values)
        else None
    )
    expected_retention = coordinate_worst["cost_retention"]
    expected_productivity = coordinate_worst["capital_productivity_per_margin_day"]
    if (
        (minimum_retention is None) != (expected_retention is None)
        or (
            minimum_retention is not None
            and not math.isclose(
                round(minimum_retention, 12),
                float(expected_retention),
                rel_tol=SUMMARY_METRIC_TOLERANCE,
                abs_tol=SUMMARY_METRIC_TOLERANCE,
            )
        )
        or (minimum_productivity is None) != (expected_productivity is None)
        or (
            minimum_productivity is not None
            and not math.isclose(
                round(minimum_productivity, 12),
                float(expected_productivity),
                rel_tol=SUMMARY_METRIC_TOLERANCE,
                abs_tol=SUMMARY_METRIC_TOLERANCE,
            )
        )
    ):
        raise CandidateLineageError(
            "candidate coordinate worst diverges from per-path/cell summaries"
        )

    pair_hhi = float(coordinate_worst["pair_hhi"])
    effective_pairs = float(coordinate_worst["effective_positive_pairs"])
    if (
        float(coordinate_worst["realized_max_drawdown_fraction"]) < 0.0
        or float(coordinate_worst["normal_effective_drawdown_fraction"]) < 0.0
        or float(coordinate_worst["stress_effective_drawdown_fraction"]) < 0.0
        or float(coordinate_worst["peak_margin_usage_fraction"]) < 0.0
        or not 0.0 <= float(coordinate_worst["margin_reject_rate"]) <= 1.0
        or not 0.0 <= float(coordinate_worst["pair_positive_share"]) <= 1.0
        or not 0.0 < pair_hhi <= 1.0
        or not math.isclose(
            round(1.0 / pair_hhi, 12),
            effective_pairs,
            rel_tol=SUMMARY_METRIC_TOLERANCE,
            abs_tol=SUMMARY_METRIC_TOLERANCE,
        )
    ):
        raise CandidateLineageError("candidate coordinate summary domain is invalid")

    visible_gate_conditions = {
        "NON_POSITIVE_COORDINATE_WORST_NET": (
            float(coordinate_worst["terminal_net_jpy"]) <= 0.0
        ),
        "NORMAL_DRAWDOWN_TOO_HIGH": (
            float(coordinate_worst["normal_effective_drawdown_fraction"])
            > float(threshold["normal_mtm_drawdown_max"])
        ),
        "STRESS_DRAWDOWN_TOO_HIGH": (
            float(coordinate_worst["stress_effective_drawdown_fraction"])
            > float(threshold["stress_mtm_drawdown_max"])
        ),
        "PEAK_MARGIN_USAGE_TOO_HIGH": (
            float(coordinate_worst["peak_margin_usage_fraction"])
            > float(threshold["peak_margin_usage_max"])
        ),
        "MARGIN_REJECT_RATE_TOO_HIGH": (
            float(coordinate_worst["margin_reject_rate"])
            > float(threshold["margin_reject_rate_max"])
        ),
        "COST_RETENTION_TOO_LOW": (
            minimum_retention is None
            or minimum_retention < float(threshold["cost_retention_min"])
        ),
        "PAIR_POSITIVE_CONTRIBUTION_TOO_CONCENTRATED": (
            float(coordinate_worst["pair_positive_share"])
            > float(threshold["pair_positive_share_max"])
        ),
        "PAIR_CONTRIBUTION_HHI_TOO_HIGH": (pair_hhi > float(threshold["pair_hhi_max"])),
        "LEAVE_ONE_PAIR_OUT_NOT_POSITIVE": (
            float(coordinate_worst["leave_one_pair_out_net_jpy"]) <= 0.0
        ),
        "CAPITAL_LOCK_METRIC_INCOMPLETE": minimum_productivity is None,
    }
    blocker_set = set(gate_blockers)
    for blocker, condition in visible_gate_conditions.items():
        if (blocker in blocker_set) != condition:
            raise CandidateLineageError(
                f"candidate summary and gate blocker diverged: {blocker}"
            )

    allowed_hidden_metric_blockers = {
        "TERMINAL_EXPOSURE",
        "MARGIN_CLOSEOUT_OCCURRED",
        "ZERO_FILLS_IN_FIXED_CELL",
    }
    allowed = {
        *mapping["risk_policy_receipt"]["blocker_codes"],
        "RUNNER_CELL_FAILURE",
        "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE",
        "COUNTERFACTUAL_LOPO_INCOMPLETE",
        *visible_gate_conditions,
        *allowed_hidden_metric_blockers,
    }
    if not blocker_set.issubset(allowed):
        raise CandidateLineageError("candidate evaluation carries an unknown blocker")

    if mapping["diagnostic_rank_eligible"] is not True:
        return
    pair_count = len(study["trade_pairs"])
    worst_return = float(coordinate_worst["terminal_net_jpy"]) / float(
        study["initial_balance_jpy"]
    )
    stress_drawdown = float(coordinate_worst["stress_effective_drawdown_fraction"])
    return_quality = _clip(worst_return / max(stress_drawdown, 0.01))
    risk_quality = 0.5 * _headroom(
        float(coordinate_worst["normal_effective_drawdown_fraction"]),
        float(threshold["normal_mtm_drawdown_max"]),
    ) + 0.5 * _headroom(
        stress_drawdown,
        float(threshold["stress_mtm_drawdown_max"]),
    )
    margin_quality = _headroom(
        float(coordinate_worst["peak_margin_usage_fraction"]),
        float(threshold["peak_margin_usage_max"]),
    )
    reject_quality = _headroom(
        float(coordinate_worst["margin_reject_rate"]),
        float(threshold["margin_reject_rate_max"]),
    )
    retention_floor = float(threshold["cost_retention_min"])
    cost_quality = _clip(
        (float(minimum_retention) - retention_floor) / max(1.0 - retention_floor, 1e-12)
    )
    equal_share = 1.0 / pair_count
    share_span = float(threshold["pair_positive_share_max"]) - equal_share
    hhi_span = float(threshold["pair_hhi_max"]) - equal_share
    share_quality = (
        _clip(
            (
                float(threshold["pair_positive_share_max"])
                - float(coordinate_worst["pair_positive_share"])
            )
            / share_span
        )
        if share_span > 0.0
        else 1.0
    )
    hhi_quality = (
        _clip((float(threshold["pair_hhi_max"]) - pair_hhi) / hhi_span)
        if hhi_span > 0.0
        else 1.0
    )
    concentration_quality = 0.5 * share_quality + 0.5 * hhi_quality
    productivity_quality = (
        _clip(float(minimum_productivity) / (1.0 + float(minimum_productivity)))
        if float(minimum_productivity) > 0.0
        else 0.0
    )
    expected_score = round(
        100.0
        * (
            0.25 * return_quality
            + 0.20 * risk_quality
            + 0.15 * margin_quality
            + 0.10 * reject_quality
            + 0.10 * cost_quality
            + 0.10 * concentration_quality
            + 0.10 * productivity_quality
        ),
        6,
    )
    if not math.isclose(
        float(mapping["diagnostic_score"]),
        expected_score,
        rel_tol=0.0,
        abs_tol=SUMMARY_SCORE_TOLERANCE,
    ):
        raise CandidateLineageError(
            "candidate diagnostic score is not derivable from its summary"
        )


def _open_events_directory(path: Path, *, create: bool) -> int:
    candidate = path.absolute()
    if candidate == candidate.parent:
        raise CandidateLineageError("events directory cannot be a filesystem root")
    try:
        state = candidate.lstat()
    except FileNotFoundError:
        if not create:
            raise CandidateLineageError("candidate lineage events directory is absent")
        parent_fd = _open_real_directory(candidate.parent, "events parent")
        try:
            try:
                os.mkdir(candidate.name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            os.fsync(parent_fd)
        except OSError as exc:
            raise CandidateLineageError(
                f"cannot create candidate lineage events directory: {exc}"
            ) from exc
        finally:
            os.close(parent_fd)
        state = candidate.lstat()
    except OSError as exc:
        raise CandidateLineageError(f"cannot stat events directory: {exc}") from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise CandidateLineageError("events directory must be a real directory")
    return _open_real_directory(candidate, "events directory")


def _open_real_directory(path: Path, label: str) -> int:
    try:
        expected = path.lstat()
    except OSError as exc:
        raise CandidateLineageError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISDIR(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
        raise CandidateLineageError(f"{label} must be a real directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        actual = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise CandidateLineageError(f"cannot open {label}: {exc}") from exc
    assert descriptor is not None
    if (
        not stat.S_ISDIR(actual.st_mode)
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
    ):
        os.close(descriptor)
        raise CandidateLineageError(f"{label} changed while being opened")
    return descriptor


def _event_names(directory_fd: int) -> list[str]:
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise CandidateLineageError(f"cannot list events directory: {exc}") from exc
    if any(_EVENT_NAME.fullmatch(name) is None for name in names):
        raise CandidateLineageError("events directory contains an unexpected file")
    return names


def _read_event(directory_fd: int, name: str) -> tuple[dict[str, Any], int]:
    try:
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise CandidateLineageError(f"cannot stat event {name}: {exc}") from exc
    if not stat.S_ISREG(state.st_mode) or state.st_size <= 0:
        raise CandidateLineageError("event must be a nonempty regular file")
    if state.st_size > MAX_EVENT_BYTES:
        raise CandidateLineageError("event byte limit exceeded")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read(MAX_EVENT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise CandidateLineageError(f"cannot read event {name}: {exc}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != state.st_dev
        or before.st_ino != state.st_ino
        or len(raw) != before.st_size
    ):
        raise CandidateLineageError("event changed while being read")
    value = _strict_event_json(raw)
    if raw != _canonical_bytes(value) + b"\n":
        raise CandidateLineageError("event bytes are not canonical JSON")
    return value, len(raw)


def _write_event_exclusive(
    directory_fd: int, name: str, event: Mapping[str, Any]
) -> None:
    payload = _canonical_bytes(event) + b"\n"
    if len(payload) > MAX_EVENT_BYTES:
        raise CandidateLineageError("event byte limit exceeded")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        handle = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = None
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise CandidateLineageError(
            "event slot already exists; reload from the current tip"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_event_json(raw: bytes) -> dict[str, Any]:
    return _strict_json(raw, reject_floats=True, label="candidate lineage event")


def _strict_artifact_json(raw: bytes) -> dict[str, Any]:
    return _strict_json(raw, reject_floats=False, label="bound artifact")


def _strict_json(raw: bytes, *, reject_floats: bool, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise CandidateLineageError(f"non-finite JSON is forbidden: {token}")

    def reject_float(token: str) -> None:
        raise CandidateLineageError(f"event JSON floating point is forbidden: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateLineageError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    kwargs: dict[str, Any] = {
        "object_pairs_hook": reject_duplicates,
        "parse_constant": reject_constant,
    }
    if reject_floats:
        kwargs["parse_float"] = reject_float
    try:
        value = json.loads(raw.decode("utf-8"), **kwargs)
    except CandidateLineageError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateLineageError(f"strict {label} JSON parse failed") from exc
    return _mapping(value, label)


def _artifact_relpath(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise CandidateLineageError("artifact relpath must be bounded text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.parts in {(), (".",)}
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise CandidateLineageError("artifact relpath must be safe and canonical")
    return value


def _exact_mapping(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    body = _mapping(value, label)
    if set(body) != keys:
        missing = sorted(keys - set(body))
        extra = sorted(set(body) - keys)
        raise CandidateLineageError(
            f"{label} shape drifted: missing={missing}, extra={extra}"
        )
    return body


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateLineageError(f"{label} must be an object")
    return _snapshot(value)


def _snapshot(value: Any) -> Any:
    try:
        return json.loads(_canonical_bytes(value))
    except (TypeError, ValueError) as exc:
        raise CandidateLineageError("value is not strict JSON") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _required_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise CandidateLineageError(f"{label} must be a lowercase sha256")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CandidateLineageError(f"{label} is invalid")
    return value


def _prefix(value: Any) -> str:
    prefix = _identifier(value, "lineage_prefix")
    if len(prefix) > 64:
        raise CandidateLineageError("lineage_prefix is too long")
    return prefix


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CandidateLineageError(f"{label} must be nonempty text")
    text = value.strip()
    if not text or len(text) > 2048 or any(ord(character) < 32 for character in text):
        raise CandidateLineageError(f"{label} must be bounded nonempty text")
    return text


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CandidateLineageError(f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateLineageError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CandidateLineageError(f"{label} must be a finite number")
    return normalized


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _headroom(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return _clip((maximum - value) / maximum)


def _unique_text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CandidateLineageError(f"{label} must be a nonempty-text list")
    if len(value) != len(set(value)):
        raise CandidateLineageError(f"{label} contains duplicates")
    return list(value)


def _identifier_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise CandidateLineageError(f"{label} must be a list")
    result = [_identifier(item, label) for item in value]
    if len(result) != len(set(result)):
        raise CandidateLineageError(f"{label} contains duplicates")
    return result


def _utc_text(value: datetime | str | Any, label: str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise CandidateLineageError(f"{label} must be timezone-aware")
        return _iso(value.astimezone(timezone.utc))
    if not isinstance(value, str):
        raise CandidateLineageError(f"{label} must be exact UTC text")
    parsed = _parse_utc(value, label)
    normalized = _iso(parsed)
    if value != normalized:
        raise CandidateLineageError(f"{label} must use canonical UTC Z notation")
    return normalized


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CandidateLineageError(f"{label} must be exact UTC Z text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CandidateLineageError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise CandidateLineageError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "CandidateLineageError",
    "CandidateLineageSnapshot",
    "EVENT_CONTRACT",
    "MAX_ATTEMPTS",
    "MAX_UNIQUE_CONFIGS",
    "MAX_UNIQUE_PROPOSALS",
    "STATUS_CONTRACT",
    "bind_result",
    "initialize_registry",
    "seal_study_attempt",
    "status_artifact",
    "verify_registry",
]
