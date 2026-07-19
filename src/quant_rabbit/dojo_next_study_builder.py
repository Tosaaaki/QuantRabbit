"""Deterministically materialize the next sealed DOJO TRAIN study.

The builder consumes an already-verified AI tuning state plus the exact model
request/response bytes recorded by that state.  A model request must itself be
one complete ``QR_DOJO_AI_TRAINER_PACKET_V1`` artifact.  The builder never
reads lineage, Drive, replay output, broker state, or a model service.

``QR_DOJO_BOT_TRAINER_STUDY_V1`` can retain only one raw-response SHA-256.  To
avoid inventing an aggregate which is not the hash of any raw response, every
candidate in one materialized study must therefore come from the same model
invocation.  One response may still contain the complete bounded candidate
set.  A future multi-response study needs a versioned trainer-study contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_rabbit.dojo_ai_trainer_packet import (
    PACKET_CONTRACT,
    DojoAITrainerPacketError,
    verify_trainer_packet,
)
from quant_rabbit.dojo_ai_tuning_state import (
    MAX_ATTEMPTS,
    MAX_PROPOSAL_SLOTS,
    MAX_STATE_STORE_EVENT_BYTES,
    DojoAITuningStateError,
    fixed_envelope_from_sealed_study,
    verify_tuning_state,
)
from quant_rabbit.dojo_bot_catalog import (
    DojoBotCatalogError,
    bot_config_risk_vector,
    catalog_manifest,
)
from quant_rabbit.dojo_bot_trainer import (
    PROPOSAL_CONTRACT,
    STUDY_CONTRACT,
    DojoBotTrainerError,
    seal_candidate_proposal,
    seal_study,
)


BUILDER_CONTRACT: Final = "QR_DOJO_NEXT_STUDY_BUILDER_V1"
PROMPT_ARTIFACT_CONTRACT: Final = "QR_DOJO_AI_TRAINER_PROMPT_ARTIFACT_V1"
UNVERIFIED_MODEL_CLAIM: Final = "unverified-caller-prompt-model"
UNVERIFIED_PROVIDER_ATTESTATION: Final = "UNVERIFIED"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_AUTHORITY = {
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}
_PROMPT_ARTIFACT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "classification",
        "trainer_packet_sha256",
        "model_request_sha256",
        "prompt_text",
        "model_claim",
        "provider_attestation",
        *_AUTHORITY,
    }
)


class DojoNextStudyBuilderError(ValueError):
    """The tuning state or proposer artifacts cannot form one exact study."""


def materialize_next_study(
    tuning_state: Mapping[str, Any],
    *,
    request_artifacts: Mapping[str, bytes],
    response_artifacts: Mapping[str, bytes],
    prompt_artifact: bytes,
) -> dict[str, Any]:
    """Build the exact ``seal_study`` artifact for the active AI attempt.

    ``request_artifacts`` and ``response_artifacts`` are keyed by invocation
    id.  Their raw byte SHA-256 values must equal the reservations in the
    tuning state.  Every request must decode to the same verified trainer
    packet, and every accepted submission must be found byte-for-byte in the
    single response whose SHA becomes ``proposer_evidence.raw_response_sha256``.
    ``prompt_artifact`` is canonical JSON bound to that sole packet and request,
    but remains explicitly a caller-supplied unverified artifact.  Its fixed
    model claim and provider attestation prevent the local hash from being
    presented as verified provenance.
    """

    try:
        state = verify_tuning_state(tuning_state)
    except DojoAITuningStateError as exc:
        raise DojoNextStudyBuilderError("AI tuning state is invalid") from exc
    if state["phase"] != "COLLECTING_PROPOSALS":
        raise DojoNextStudyBuilderError(
            "next study requires a complete COLLECTING_PROPOSALS attempt"
        )
    attempt = _active_attempt(state)
    if len(attempt["invocations"]) != 1:
        raise DojoNextStudyBuilderError(
            "REVIEW_REQUIRED_STUDY_V1_SINGLE_MODEL_INVOCATION_REQUIRED"
        )
    if attempt["dispatch"] is not None or attempt["terminal"] is not None:
        raise DojoNextStudyBuilderError(
            "next study must be materialized before dispatch or terminal state"
        )
    if any(row["response_sha256"] is None for row in attempt["invocations"]):
        raise DojoNextStudyBuilderError("model invocation response is incomplete")

    _verify_artifact_keyset(request_artifacts, attempt["invocations"], label="request")
    _verify_artifact_keyset(
        response_artifacts, attempt["invocations"], label="response"
    )
    packet = _verify_request_packets(state, attempt, request_artifacts)
    accepted = _accepted_submissions(attempt)
    if not accepted:
        raise DojoNextStudyBuilderError("active attempt has no accepted proposals")

    accepted_invocation_ids = {
        invocation["invocation_id"]
        for invocation in attempt["invocations"]
        if any(row["status"] == "ACCEPTED" for row in invocation["submissions"])
    }
    if len(accepted_invocation_ids) != 1:
        raise DojoNextStudyBuilderError(
            "study V1 cannot bind accepted proposals from multiple raw responses"
        )
    accepted_invocation_id = next(iter(accepted_invocation_ids))
    _verify_response_artifacts(attempt, response_artifacts)
    _verify_accepted_response_membership(
        attempt,
        accepted_invocation_id=accepted_invocation_id,
        response_raw=response_artifacts[accepted_invocation_id],
    )

    _verify_search_budgets(state, attempt, packet, accepted)
    _verify_global_identities(state, accepted)
    candidates = _candidate_denominator(accepted)
    envelope = state["fixed_envelope"]
    _require_nonincreasing_bounded_risk(
        candidates,
        packet=packet,
        stress_slippage_pips_per_fill=envelope["cost_arms"]["STRESS"][
            "slippage_pips_per_fill"
        ],
    )
    accepted_invocation = next(
        row
        for row in attempt["invocations"]
        if row["invocation_id"] == accepted_invocation_id
    )
    prompt_sha256 = _verify_unverified_prompt_artifact(
        prompt_artifact,
        trainer_packet_sha256=packet["packet_sha256"],
        model_request_sha256=accepted_invocation["request_sha256"],
    )
    study_id = _study_id(
        state,
        attempt_ordinal=attempt["attempt_ordinal"],
        packet_sha256=packet["packet_sha256"],
        response_sha256=accepted_invocation["response_sha256"],
    )
    study = {
        "contract": STUDY_CONTRACT,
        "schema_version": 1,
        "study_id": study_id,
        "window_role": envelope["window_role"],
        "initial_balance_jpy": envelope["initial_balance_jpy"],
        "trade_pairs": list(envelope["trade_pairs"]),
        "feed_pairs": list(envelope["feed_pairs"]),
        "candidates": candidates,
        "thresholds": _clone(envelope["thresholds"]),
        "window": _clone(envelope["window"]),
        "cost_arms": _clone(envelope["cost_arms"]),
        "proposer_evidence": {
            "prompt_sha256": prompt_sha256,
            "input_sha256": accepted_invocation["request_sha256"],
            "raw_response_sha256": accepted_invocation["response_sha256"],
            "model_claim": UNVERIFIED_MODEL_CLAIM,
            "provider_attestation": UNVERIFIED_PROVIDER_ATTESTATION,
        },
        "search_budget": {
            "attempt_ordinal": attempt["attempt_ordinal"],
            "total_attempts_in_lineage": MAX_ATTEMPTS,
            "max_candidates": len(candidates),
        },
    }
    try:
        sealed = seal_study(study, envelope["source_digests"])
    except DojoBotTrainerError as exc:
        raise DojoNextStudyBuilderError(
            "accepted proposals cannot form a valid sealed study"
        ) from exc
    if fixed_envelope_from_sealed_study(sealed) != envelope:
        raise DojoNextStudyBuilderError("materialized study changed the fixed envelope")
    if [row["proposal_sha256"] for row in sealed["study"]["candidates"]] != [
        row["proposal_sha256"] for row in candidates
    ]:
        raise DojoNextStudyBuilderError(
            "materialized study changed the accepted candidate denominator"
        )
    for key, expected in _AUTHORITY.items():
        if sealed[key] != expected:
            raise DojoNextStudyBuilderError("materialized study gained authority")
    return sealed


def _verify_request_packets(
    state: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    packet: dict[str, Any] | None = None
    for invocation in attempt["invocations"]:
        invocation_id = invocation["invocation_id"]
        raw = _bytes(artifacts[invocation_id], f"request {invocation_id}")
        if hashlib.sha256(raw).hexdigest() != invocation["request_sha256"]:
            raise DojoNextStudyBuilderError("model request SHA-256 drifted")
        value = _strict_json(raw, label=f"request {invocation_id}")
        if not isinstance(value, Mapping):
            raise DojoNextStudyBuilderError("model request is not a trainer packet")
        try:
            verified = verify_trainer_packet(value)
        except DojoAITrainerPacketError as exc:
            raise DojoNextStudyBuilderError(
                "model request trainer packet is invalid"
            ) from exc
        if packet is None:
            packet = verified
        elif verified != packet:
            raise DojoNextStudyBuilderError(
                "model invocations used different trainer packets"
            )
    assert packet is not None
    _verify_packet_bindings(state, attempt, packet)
    return packet


def _verify_packet_bindings(
    state: Mapping[str, Any],
    attempt: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> None:
    if packet["contract"] != PACKET_CONTRACT:
        raise DojoNextStudyBuilderError("trainer packet contract drifted")
    source = packet["source_bindings"]
    prior = attempt["prior_result_binding"]
    expected_source = {
        "registry_id": state["registry_id"],
        "lineage_prefix": state["lineage_prefix"],
        "attempt_ordinal": prior["attempt_ordinal"],
        "study_sha256": prior["study_sha256"],
        "evaluation_sha256": prior["evaluation_sha256"],
        "evaluation_artifact_sha256": prior["evaluation_artifact_sha256"],
        "lineage_result_event_sha256": prior["result_event_sha256"],
        "lineage_tip_sha256": prior["lineage_tip_sha256"],
        "fixed_envelope_sha256": state["fixed_envelope_sha256"],
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise DojoNextStudyBuilderError(
                f"trainer packet source binding drifted: {key}"
            )
    if source.get("exact_result_binding_verified") is not True:
        raise DojoNextStudyBuilderError("trainer packet lacks exact result binding")
    _sha(source.get("tuning_state_sha256"), "packet tuning state SHA-256")

    envelope = state["fixed_envelope"]
    expected_environment = {
        "window_role": envelope["window_role"],
        "window": envelope["window"],
        "initial_balance_jpy": envelope["initial_balance_jpy"],
        "trade_pairs": envelope["trade_pairs"],
        "feed_pairs": envelope["feed_pairs"],
        "intrabar_paths": envelope["intrabar_paths"],
        "cost_arms": envelope["cost_arms"],
        "thresholds": envelope["thresholds"],
    }
    fixed_environment = packet["fixed_environment"]
    for key, expected in expected_environment.items():
        if fixed_environment.get(key) != expected:
            raise DojoNextStudyBuilderError(
                f"trainer packet fixed environment drifted: {key}"
            )
    if fixed_environment.get("catalog") != catalog_manifest():
        raise DojoNextStudyBuilderError("trainer packet catalog binding drifted")


def _verify_search_budgets(
    state: Mapping[str, Any],
    attempt: Mapping[str, Any],
    packet: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
) -> None:
    ordinal = attempt["attempt_ordinal"]
    if ordinal > MAX_ATTEMPTS or state["max_attempts"] != MAX_ATTEMPTS:
        raise DojoNextStudyBuilderError("attempt budget is exhausted or drifted")
    prior_attempts = (
        int(state["initial_attempts_consumed"]) + len(state["attempts"]) - 1
    )
    prior_slots = int(state["initial_proposal_slots_consumed"]) + sum(
        int(invocation["proposal_slot_charge"])
        for prior_attempt in state["attempts"][:-1]
        for invocation in prior_attempt["invocations"]
    )
    current_charge = sum(
        int(invocation["proposal_slot_charge"]) for invocation in attempt["invocations"]
    )
    consumed = prior_slots + current_charge
    if consumed > MAX_PROPOSAL_SLOTS or len(accepted) > current_charge:
        raise DojoNextStudyBuilderError("proposal budget is exhausted or inconsistent")
    budget = packet["search_budget"]
    expected = {
        "phase": "READY_FOR_MODEL",
        "attempts_consumed": prior_attempts,
        "attempts_remaining": MAX_ATTEMPTS - prior_attempts,
        "max_attempts": MAX_ATTEMPTS,
        "proposal_slots_consumed": prior_slots,
        "proposal_slots_remaining": MAX_PROPOSAL_SLOTS - prior_slots,
        "max_proposal_slots": MAX_PROPOSAL_SLOTS,
        "invalid_proposal_count": sum(
            submission["status"] in {"INVALID", "BUDGET_REJECTED"}
            for prior_attempt in state["attempts"][:-1]
            for invocation in prior_attempt["invocations"]
            for submission in invocation["submissions"]
        ),
        "duplicate_proposal_count": sum(
            submission["status"] == "DUPLICATE"
            for prior_attempt in state["attempts"][:-1]
            for invocation in prior_attempt["invocations"]
            for submission in invocation["submissions"]
        ),
    }
    for key, value in expected.items():
        if budget.get(key) != value:
            raise DojoNextStudyBuilderError(
                f"trainer packet search budget drifted: {key}"
            )
    if ordinal != prior_attempts + 1:
        raise DojoNextStudyBuilderError("attempt ordinal is not the next budget slot")


def _verify_global_identities(
    state: Mapping[str, Any], accepted: Sequence[Mapping[str, Any]]
) -> None:
    identities = [row["executable_identity_sha256"] for row in accepted]
    if any(identity is None for identity in identities):
        raise DojoNextStudyBuilderError("accepted proposal lacks executable identity")
    if len(identities) != len(set(identities)):
        raise DojoNextStudyBuilderError("accepted executable identity is duplicated")
    global_identities = list(state["global_executable_identity_sha256s"])
    if len(global_identities) != len(set(global_identities)):
        raise DojoNextStudyBuilderError(
            "global executable identity registry is duplicated"
        )
    accepted_set = set(identities)
    if not accepted_set.issubset(global_identities):
        raise DojoNextStudyBuilderError("accepted identity is not globally burned")
    burned_before = set(global_identities) - accepted_set
    if accepted_set & burned_before:
        raise DojoNextStudyBuilderError(
            "accepted identity repeats an earlier executable config"
        )


def _verify_response_artifacts(
    attempt: Mapping[str, Any], artifacts: Mapping[str, bytes]
) -> None:
    for invocation in attempt["invocations"]:
        invocation_id = invocation["invocation_id"]
        raw = _bytes(artifacts[invocation_id], f"response {invocation_id}")
        if hashlib.sha256(raw).hexdigest() != invocation["response_sha256"]:
            raise DojoNextStudyBuilderError("model response SHA-256 drifted")


def _verify_accepted_response_membership(
    attempt: Mapping[str, Any],
    *,
    accepted_invocation_id: str,
    response_raw: bytes,
) -> None:
    value = _strict_json(response_raw, label="accepted model response")
    if not isinstance(value, list):
        raise DojoNextStudyBuilderError("accepted model response is not an array")
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        submission_id = item.get("submission_id")
        if not isinstance(submission_id, str):
            continue
        if submission_id in raw_by_id:
            raise DojoNextStudyBuilderError(
                "accepted model response repeats a submission id"
            )
        raw_by_id[submission_id] = item
    invocation = next(
        row
        for row in attempt["invocations"]
        if row["invocation_id"] == accepted_invocation_id
    )
    accepted = [row for row in invocation["submissions"] if row["status"] == "ACCEPTED"]
    for stored in accepted:
        raw = raw_by_id.get(stored["submission_id"])
        if raw is None or set(raw) != {
            "submission_id",
            "raw_proposal_sha256",
            "proposal",
            "validation_errors",
        }:
            raise DojoNextStudyBuilderError(
                "accepted submission is absent from its raw response"
            )
        if raw["validation_errors"] != []:
            raise DojoNextStudyBuilderError(
                "accepted raw submission carries validation errors"
            )
        proposal = raw["proposal"]
        raw_proposal_sha = _canonical_sha(proposal)
        if (
            raw.get("raw_proposal_sha256") != raw_proposal_sha
            or stored["raw_proposal_sha256"] != raw_proposal_sha
        ):
            raise DojoNextStudyBuilderError("raw proposal SHA-256 is not exact")
        try:
            sealed = seal_candidate_proposal(proposal)
        except DojoBotTrainerError as exc:
            raise DojoNextStudyBuilderError(
                "accepted raw proposal is no longer catalog-valid"
            ) from exc
        if sealed != stored["sealed_proposal"]:
            raise DojoNextStudyBuilderError(
                "accepted proposal differs from the raw model response"
            )


def _candidate_denominator(
    accepted: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [seal_candidate_proposal(row["sealed_proposal"]) for row in accepted]
    candidates.sort(key=lambda row: row["candidate_id"])
    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise DojoNextStudyBuilderError("accepted candidate id is duplicated")
    if any(row["contract"] != PROPOSAL_CONTRACT for row in candidates):
        raise DojoNextStudyBuilderError("accepted proposal contract drifted")
    if any(row["risk_increase"] is not False for row in candidates):
        raise DojoNextStudyBuilderError("accepted proposal increases risk")
    return candidates


def _require_nonincreasing_bounded_risk(
    candidates: Sequence[Mapping[str, Any]],
    *,
    packet: Mapping[str, Any],
    stress_slippage_pips_per_fill: float,
) -> None:
    """Require an absolute-safe candidate dominated by a prior same-family row.

    Catalog validity alone deliberately allows a wider TRAIN search surface
    than the repo-owned rank-admission policy.  The next-attempt materializer
    is narrower: it permits only bounded fixed-stop candidates inside that
    absolute policy, and only when every bounded same-family candidate in the
    immediately preceding current-run packet has equal-or-greater leverage,
    concurrency, holding ceiling, stop distance, and stop exposure.  Older
    ``previous_attempts`` are never eligible parents for a new materialization.
    Legacy ``max_concurrent`` is normalized by ``bot_config_risk_vector`` into
    ``max_concurrent_per_pair`` before this comparison.
    """

    current_families, parents_by_family = _current_run_risk_parents(
        packet,
        stress_slippage_pips_per_fill=stress_slippage_pips_per_fill,
    )
    for candidate in candidates:
        try:
            vector = bot_config_risk_vector(
                candidate["config"],
                stress_slippage_pips_per_fill=stress_slippage_pips_per_fill,
            )
        except DojoBotCatalogError as exc:
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_CANDIDATE_RISK_VECTOR_INVALID"
            ) from exc
        if vector["config_sha256"] != candidate["config_sha256"]:
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_CANDIDATE_RISK_IDENTITY_DRIFT"
            )
        if vector["hard_envelope_passed"] is not True:
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_ABSOLUTE_RISK_HARD_ENVELOPE_EXCEEDED"
            )
        if vector["rankable"] is not True or not _has_bounded_fixed_stop(vector):
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_BOUNDED_INITIAL_STOP_REQUIRED"
            )
        vector["ceiling_min"] = candidate["config"]["ceiling_min"]
        family = candidate["family"]
        parents = parents_by_family.get(family, [])
        if family not in current_families:
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_NEW_FAMILY_BASELINE_REVISION_REQUIRED"
            )
        if not parents:
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_SAME_FAMILY_BOUNDED_PARENT_REQUIRED"
            )
        if not all(_risk_vector_dominates(parent, vector) for parent in parents):
            raise DojoNextStudyBuilderError("REVIEW_REQUIRED_RELATIVE_RISK_INCREASE")


def _current_run_risk_parents(
    packet: Mapping[str, Any],
    *,
    stress_slippage_pips_per_fill: float,
) -> tuple[set[str], dict[str, list[dict[str, Any]]]]:
    current = packet.get("candidates")
    if not isinstance(current, list):
        raise DojoNextStudyBuilderError(
            "REVIEW_REQUIRED_PRIOR_CANDIDATE_DENOMINATOR_INVALID"
        )
    result: dict[str, list[dict[str, Any]]] = {}
    families: set[str] = set()
    seen_ids: set[str] = set()
    for row in current:
        if not isinstance(row, Mapping):
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_PRIOR_CANDIDATE_DENOMINATOR_INVALID"
            )
        candidate_id = row.get("candidate_id")
        family = row.get("family")
        config = row.get("config")
        config_sha256 = row.get("config_sha256")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in seen_ids
            or not isinstance(family, str)
            or not isinstance(config, Mapping)
            or not isinstance(config_sha256, str)
        ):
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_PRIOR_CANDIDATE_DENOMINATOR_INVALID"
            )
        seen_ids.add(candidate_id)
        families.add(family)
        try:
            vector = bot_config_risk_vector(
                config,
                stress_slippage_pips_per_fill=stress_slippage_pips_per_fill,
            )
        except DojoBotCatalogError as exc:
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_PRIOR_CANDIDATE_RISK_VECTOR_INVALID"
            ) from exc
        if vector["signal"] != family or vector["config_sha256"] != config_sha256:
            raise DojoNextStudyBuilderError(
                "REVIEW_REQUIRED_PRIOR_CANDIDATE_RISK_IDENTITY_DRIFT"
            )
        if _has_bounded_fixed_stop(vector):
            vector["ceiling_min"] = config["ceiling_min"]
            result.setdefault(family, []).append(vector)
    return families, result


def _has_bounded_fixed_stop(vector: Mapping[str, Any]) -> bool:
    return (
        vector.get("initial_stop_bound_kind") == "FIXED_SL_PIPS"
        and vector.get("initial_sl_pips") is not None
        and vector.get("single_stop_risk_index") is not None
        and vector.get("gross_stop_risk_index") is not None
    )


def _risk_vector_dominates(
    parent: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    """Return whether candidate risk is componentwise no greater than parent."""

    comparable_fields = (
        "per_position_leverage",
        "max_concurrent_per_pair",
        "max_global_concurrent",
        "pair_gross_leverage",
        "global_gross_leverage",
        "initial_sl_pips",
        "single_stop_risk_index",
        "gross_stop_risk_index",
        "ceiling_min",
    )
    return all(candidate[field] <= parent[field] for field in comparable_fields)


def _active_attempt(state: Mapping[str, Any]) -> dict[str, Any]:
    attempts = state["attempts"]
    if not attempts:
        raise DojoNextStudyBuilderError("tuning state has no active attempt")
    return attempts[-1]


def _accepted_submissions(
    attempt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        row
        for invocation in attempt["invocations"]
        for row in invocation["submissions"]
        if row["status"] == "ACCEPTED"
    ]


def _verify_artifact_keyset(
    artifacts: Mapping[str, bytes],
    invocations: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if not isinstance(artifacts, Mapping):
        raise DojoNextStudyBuilderError(f"{label} artifacts must be a mapping")
    expected = {row["invocation_id"] for row in invocations}
    if set(artifacts) != expected:
        raise DojoNextStudyBuilderError(
            f"{label} artifact invocation denominator is incomplete"
        )


def _study_id(
    state: Mapping[str, Any],
    *,
    attempt_ordinal: int,
    packet_sha256: str,
    response_sha256: str,
) -> str:
    suffix = _canonical_sha(
        {
            "contract": BUILDER_CONTRACT,
            "registry_id": state["registry_id"],
            "attempt_ordinal": attempt_ordinal,
            "packet_sha256": packet_sha256,
            "response_sha256": response_sha256,
        }
    )[:16]
    value = f"dojo-ai-attempt-{attempt_ordinal}-{suffix}"
    return _identifier(value, "study_id")


def _verify_unverified_prompt_artifact(
    raw_value: bytes,
    *,
    trainer_packet_sha256: str,
    model_request_sha256: str,
) -> str:
    raw = _bytes(raw_value, "prompt artifact")
    value = _strict_json(raw, label="prompt artifact")
    if not isinstance(value, Mapping) or set(value) != _PROMPT_ARTIFACT_KEYS:
        raise DojoNextStudyBuilderError("prompt artifact schema mismatch")
    if (
        value["contract"] != PROMPT_ARTIFACT_CONTRACT
        or value["schema_version"] != 1
        or value["classification"] != "UNVERIFIED_CALLER_ARTIFACT"
        or value["trainer_packet_sha256"] != trainer_packet_sha256
        or value["model_request_sha256"] != model_request_sha256
        or value["model_claim"] != UNVERIFIED_MODEL_CLAIM
        or value["provider_attestation"] != UNVERIFIED_PROVIDER_ATTESTATION
        or not isinstance(value["prompt_text"], str)
        or not value["prompt_text"].strip()
    ):
        raise DojoNextStudyBuilderError(
            "prompt artifact provenance or request binding drifted"
        )
    for key, expected in _AUTHORITY.items():
        if value[key] != expected:
            raise DojoNextStudyBuilderError("prompt artifact gained authority")
    if raw != _canonical_json_bytes(value) + b"\n":
        raise DojoNextStudyBuilderError("prompt artifact is not canonical JSON")
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_constant(token: str) -> None:
        raise DojoNextStudyBuilderError(
            f"{label} contains forbidden non-finite JSON token {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoNextStudyBuilderError(
                    f"{label} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoNextStudyBuilderError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DojoNextStudyBuilderError(f"{label} is not strict JSON") from exc
    _validate_json(value, label=label)
    return value


def _validate_json(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoNextStudyBuilderError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DojoNextStudyBuilderError(f"{label} has a non-string key")
        for item in value.values():
            _validate_json(item, label=label)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _validate_json(item, label=label)
        return
    raise DojoNextStudyBuilderError(f"{label} is not strict JSON")


def _bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise DojoNextStudyBuilderError(f"{label} must be nonempty bytes")
    if len(value) > MAX_STATE_STORE_EVENT_BYTES:
        raise DojoNextStudyBuilderError(f"{label} exceeds the artifact size limit")
    return value


def _raw_sha(value: Any, label: str) -> str:
    return hashlib.sha256(_bytes(value, label)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DojoNextStudyBuilderError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DojoNextStudyBuilderError(f"{label} is not a bounded identifier")
    return value


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


__all__ = [
    "BUILDER_CONTRACT",
    "DojoNextStudyBuilderError",
    "PROMPT_ARTIFACT_CONTRACT",
    "UNVERIFIED_MODEL_CLAIM",
    "UNVERIFIED_PROVIDER_ATTESTATION",
    "materialize_next_study",
]
