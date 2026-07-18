"""Fail-closed, offline-only primitives for DOJO AI-discretion evaluation.

The module deliberately does not call a model, open a broker connection, or
grant live authority.  A trial has four phases: prelock prompt/scorer and a
fresh-context capability manifest, build one pseudonymized packet, seal one fixed
schema response, then (and only then) open the answer key in the scorer. Removing
the calendar date is pseudonymization, not anonymity: exact prices can reveal it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


SOURCE_CONTRACT = "QR_DOJO_AI_DAY_SOURCE_V1"
CAPABILITY_MANIFEST_CONTRACT = "QR_DOJO_AI_CAPABILITY_MANIFEST_V1"
PROMPT_LOCK_CONTRACT = "QR_DOJO_AI_PROMPT_LOCK_V1"
SCORER_LOCK_CONTRACT = "QR_DOJO_AI_SCORER_LOCK_V1"
MODEL_MANIFEST_CONTRACT = "QR_DOJO_AI_MODEL_MANIFEST_V1"
PACKET_CONTRACT = "QR_DOJO_AI_DISCRETION_PACKET_V1"
RESPONSE_CONTRACT = "QR_DOJO_AI_DISCRETION_RESPONSE_V1"
ANSWER_KEY_CONTRACT = "QR_DOJO_AI_DISCRETION_ANSWER_KEY_V1"
SCORE_CONTRACT = "QR_DOJO_AI_DISCRETION_SCORE_V1"
PILOT_SCORE_CONTRACT = "QR_DOJO_AI_DISCRETION_PILOT_SCORE_V1"
VALIDITY_REGISTRY_CONTRACT = "QR_DOJO_AI_VALIDITY_REGISTRY_V1"

VALID = "VALID"
INVALIDATED = "INVALIDATED"
DIAGNOSTIC_TIER = "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
MAX_OBSERVATIONS = 512
MAX_PAYLOAD_BYTES = 64_000

FIXED_RESPONSE_KEYS = frozenset(
    {
        "trial_id",
        "action",
        "pair",
        "size",
        "confidence",
        "evidence_refs",
        "target_pips",
        "invalidation_pips",
        "strongest_counterargument",
        "abstain_reason",
    }
)
FIXED_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(FIXED_RESPONSE_KEYS),
    "properties": {
        "trial_id": {"type": "string"},
        "action": {"enum": ["FLAT", "LONG", "SHORT"]},
        "pair": {"type": "string"},
        "size": {"enum": ["NONE", "HALF", "FULL"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        },
        "target_pips": {"type": ["number", "null"]},
        "invalidation_pips": {"type": ["number", "null"]},
        "strongest_counterargument": {"type": "string"},
        "abstain_reason": {"type": ["string", "null"]},
    },
}

_SOURCE_KEYS = {
    "contract",
    "blind_nonce",
    "pair",
    "decision_cutoff_utc",
    "observations",
}
_OBSERVATION_KEYS = {"observed_at_utc", "kind", "payload"}
_PACKET_OBSERVATION_KEYS = {"id", "seconds_before_cutoff", "kind", "payload"}
_PACKET_KEYS = {
    "contract",
    "schema_version",
    "validity_status",
    "trial_id",
    "pair",
    "decision_time_label",
    "observations",
    "source_sha256",
    "prompt_lock_sha256",
    "capability_manifest_sha256",
    "scorer_lock_sha256",
    "calendar_date_removed",
    "anonymity_status",
    "reidentification_risk",
    "single_day_packet",
    "read_only",
    "ai_order_authority",
    "live_permission",
    "broker_mutation_allowed",
    "evidence_tier",
    "external_attestations_verified",
    "attestation_gap_codes",
    "packet_sha256",
}
_INVALIDATION_METADATA_KEYS = {
    "invalidation_reason",
    "invalidates_artifact_sha256",
    "invalidation_evidence_sha256",
    "invalidated_at_utc",
    "diagnostic_status",
}
_CAPABILITY_KEYS = {
    "contract",
    "schema_version",
    "validity_status",
    "generated_at_utc",
    "context_id",
    "declared_fresh_context",
    "declared_mounted_artifact_roles",
    "declared_answer_key_physically_absent",
    "declared_answer_key_access",
    "declared_repository_access",
    "declared_filesystem_access",
    "declared_network_access",
    "declared_browser_access",
    "declared_conversation_history_access",
    "declared_persistent_memory_access",
    "declared_tools",
    "external_send_allowed",
    "model_api_invoked_by_runtime",
    "live_permission",
    "broker_mutation_allowed",
    "evidence_tier",
    "external_attestations_verified",
    "attestation_gap_codes",
    "capability_manifest_sha256",
}
_PROMPT_LOCK_KEYS = {
    "contract",
    "schema_version",
    "validity_status",
    "locked_at_utc",
    "variant_id",
    "prompt_text",
    "prompt_sha256",
    "fixed_output_schema",
    "output_schema_sha256",
    "optimization_after_lock_allowed",
    "live_permission",
    "chronology_attestation_status",
    "evidence_tier",
    "external_attestations_verified",
    "attestation_gap_codes",
    "prompt_lock_sha256",
}
_SCORER_LOCK_KEYS = {
    "contract",
    "schema_version",
    "validity_status",
    "locked_at_utc",
    "policy",
    "scorer_sha256",
    "answer_key_open_before_response_seal_allowed",
    "live_permission",
    "chronology_attestation_status",
    "evidence_tier",
    "external_attestations_verified",
    "attestation_gap_codes",
    "scorer_lock_sha256",
}
_RETURN_KEYS = {
    "FLAT",
    "LONG_HALF",
    "LONG_FULL",
    "SHORT_HALF",
    "SHORT_FULL",
}
_OWN_SHA_FIELDS = {
    CAPABILITY_MANIFEST_CONTRACT: "capability_manifest_sha256",
    PROMPT_LOCK_CONTRACT: "prompt_lock_sha256",
    SCORER_LOCK_CONTRACT: "scorer_lock_sha256",
    MODEL_MANIFEST_CONTRACT: "model_sha256",
    PACKET_CONTRACT: "packet_sha256",
    RESPONSE_CONTRACT: "response_receipt_sha256",
    ANSWER_KEY_CONTRACT: "answer_key_sha256",
    SCORE_CONTRACT: "score_receipt_sha256",
    PILOT_SCORE_CONTRACT: "pilot_score_sha256",
    VALIDITY_REGISTRY_CONTRACT: "validity_registry_sha256",
}
_DATE_TEXT = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_FORBIDDEN_PAYLOAD_KEY_PARTS = {
    "answer",
    "future",
    "forward",
    "outcome",
    "realized",
    "post_cutoff",
    "exit_price",
    "ground_truth",
    "target_label",
}
_DATE_PAYLOAD_KEYS = {
    "date",
    "day",
    "decision_date",
    "timestamp",
    "time_utc",
    "datetime",
    "observed_at_utc",
}
_SCORER_POLICY = {
    "metric": "NET_LOG_GROWTH_ALL_TRIALS",
    "flat_return": 0.0,
    "missing_or_schema_invalid_response": "REJECT_BEFORE_KEY_OPEN",
    "same_model_lineage_handling": "DECLARED_CLUSTER_ONLY_NO_INDEPENDENCE_CLAIM",
    "invalid_parent_policy": "TRANSITIVE_INVALIDATION",
    "positive_result_grants_live_permission": False,
}

_ATTESTATION_GAPS = [
    "SANDBOX_NOT_EXTERNALLY_ATTESTED",
    "MODEL_INVOCATION_NOT_PROVIDER_ATTESTED",
    "CHRONOLOGY_NOT_EXTERNALLY_MONOTONIC",
    "ANSWER_KEY_NOT_MARKET_PROVIDER_ATTESTED",
    "REGISTRY_NOT_EXTERNALLY_MONOTONIC",
]


def canonical_sha256(value: Any) -> str:
    """Hash one JSON value using the repository's canonical representation."""

    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _diagnostic_fields() -> dict[str, Any]:
    """Truthful evidence boundary for this local, public-hash-only harness."""

    return {
        "evidence_tier": DIAGNOSTIC_TIER,
        "external_attestations_verified": False,
        "attestation_gap_codes": list(_ATTESTATION_GAPS),
    }


def build_capability_manifest(
    *, context_id: str, generated_at_utc: datetime
) -> dict[str, Any]:
    """Seal the no-tools fresh context in which one packet may be evaluated."""

    context = _bounded_text(context_id, "context_id", maximum=200)
    generated = _aware_utc(generated_at_utc)
    body = {
        "contract": CAPABILITY_MANIFEST_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "generated_at_utc": generated.isoformat(),
        "context_id": context,
        "declared_fresh_context": True,
        "declared_mounted_artifact_roles": ["INLINE_PACKET_ONLY"],
        "declared_answer_key_physically_absent": True,
        "declared_answer_key_access": False,
        "declared_repository_access": False,
        "declared_filesystem_access": False,
        "declared_network_access": False,
        "declared_browser_access": False,
        "declared_conversation_history_access": False,
        "declared_persistent_memory_access": False,
        "declared_tools": [],
        "external_send_allowed": False,
        "model_api_invoked_by_runtime": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        **_diagnostic_fields(),
    }
    return _seal(body, "capability_manifest_sha256")


def prelock_prompt(
    prompt_text: str,
    *,
    variant_id: str,
    locked_at_utc: datetime,
) -> dict[str, Any]:
    """Freeze a prompt and the only accepted response schema before a trial."""

    # Prompt preregistration hashes the exact UTF-8 file bytes.  Do not call
    # ``strip()`` here: doing so used to make every honest runtime prompt hash
    # differ from its registered hash merely because the file ended in a
    # newline.
    text = _bounded_exact_text(prompt_text, "prompt_text", maximum=8_000)
    variant = _bounded_text(variant_id, "variant_id", maximum=120)
    if _DATE_TEXT.search(text):
        raise ValueError("prompt must not contain an identifiable decision date")
    lowered = text.lower()
    if "answer_key" in lowered or "answer key" in lowered:
        raise ValueError("prompt must not reference an answer key")
    locked = _aware_utc(locked_at_utc)
    body = {
        "contract": PROMPT_LOCK_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "locked_at_utc": locked.isoformat(),
        "variant_id": variant,
        "prompt_text": text,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "fixed_output_schema": FIXED_RESPONSE_SCHEMA,
        "output_schema_sha256": canonical_sha256(FIXED_RESPONSE_SCHEMA),
        "optimization_after_lock_allowed": False,
        "live_permission": False,
        "chronology_attestation_status": "SELF_ATTESTED_UNVERIFIED",
        **_diagnostic_fields(),
    }
    return _seal(body, "prompt_lock_sha256")


def prelock_scorer(*, locked_at_utc: datetime) -> dict[str, Any]:
    """Freeze scoring semantics before any response or answer is observed."""

    locked = _aware_utc(locked_at_utc)
    body = {
        "contract": SCORER_LOCK_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "locked_at_utc": locked.isoformat(),
        "policy": _SCORER_POLICY,
        "scorer_sha256": canonical_sha256(_SCORER_POLICY),
        "answer_key_open_before_response_seal_allowed": False,
        "live_permission": False,
        "chronology_attestation_status": "SELF_ATTESTED_UNVERIFIED",
        **_diagnostic_fields(),
    }
    return _seal(body, "scorer_lock_sha256")


def seal_model_manifest(
    *,
    model_name: str,
    model_version: str,
    model_lineage: str,
    reasoning_effort: str,
    context_id: str,
    capability_manifest: Mapping[str, Any],
    locked_at_utc: datetime,
) -> dict[str, Any]:
    """Bind one declared offline model identity to one fresh context."""

    capability = _snapshot(capability_manifest)
    _validate_capability_manifest(capability)
    if capability["validity_status"] != VALID:
        raise ValueError("capability manifest is invalidated")
    context = _bounded_text(context_id, "context_id", maximum=200)
    if context != capability["context_id"]:
        raise ValueError("model context does not match capability manifest")
    body = {
        "contract": MODEL_MANIFEST_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "locked_at_utc": _aware_utc(locked_at_utc).isoformat(),
        "model_name": _bounded_text(model_name, "model_name", maximum=200),
        "model_version": _bounded_text(model_version, "model_version", maximum=200),
        "declared_model_lineage": _bounded_text(
            model_lineage, "model_lineage", maximum=200
        ),
        "model_lineage_attestation_status": "DECLARED_ONLY_NOT_PROVIDER_ATTESTED",
        "reasoning_effort": _bounded_text(
            reasoning_effort, "reasoning_effort", maximum=80
        ),
        "context_id": context,
        "capability_manifest_sha256": capability["capability_manifest_sha256"],
        "declared_fresh_context": True,
        "model_api_invoked_by_runtime": False,
        "external_send_allowed": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "chronology_attestation_status": "SELF_ATTESTED_UNVERIFIED",
        **_diagnostic_fields(),
    }
    return _seal(body, "model_sha256")


def build_day_packet(
    source: Mapping[str, Any],
    *,
    prompt_lock: Mapping[str, Any],
    capability_manifest: Mapping[str, Any],
    scorer_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one calendar-date-removed packet and reject post-cutoff observations."""

    if not isinstance(source, Mapping):
        raise ValueError("single-day source must be one JSON object")
    source_value = _snapshot(source)
    if set(source_value) != _SOURCE_KEYS:
        raise ValueError("single-day source shape is invalid")
    if source_value.get("contract") != SOURCE_CONTRACT:
        raise ValueError("single-day source contract is invalid")
    blind_nonce = _require_sha256(source_value.get("blind_nonce"), "blind nonce")

    prompt = _snapshot(prompt_lock)
    capability = _snapshot(capability_manifest)
    scorer = _snapshot(scorer_lock)
    _validate_prompt_lock(prompt)
    _validate_capability_manifest(capability)
    _validate_scorer_lock(scorer)
    invalid_parents = _invalid_parent_contracts((prompt, capability, scorer))

    pair = _bounded_text(source_value.get("pair"), "pair", maximum=40)
    if not re.fullmatch(r"[A-Z]{3}_[A-Z]{3}", pair):
        raise ValueError("pair must use AAA_BBB format")
    cutoff = _parse_utc_required(
        source_value.get("decision_cutoff_utc"), "decision cutoff"
    )
    raw_observations = source_value.get("observations")
    if (
        not isinstance(raw_observations, list)
        or not 1 <= len(raw_observations) <= MAX_OBSERVATIONS
    ):
        raise ValueError("observations must be one bounded non-empty list")

    normalized: list[tuple[datetime, str, dict[str, Any]]] = []
    for index, raw in enumerate(raw_observations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"observation {index} must be an object")
        observation = _snapshot(raw)
        if set(observation) != _OBSERVATION_KEYS:
            raise ValueError(f"observation {index} shape is invalid")
        observed = _parse_utc_required(
            observation.get("observed_at_utc"), "observation timestamp"
        )
        if observed > cutoff:
            raise ValueError(f"observation {index} is after decision cutoff")
        kind = _bounded_text(observation.get("kind"), "observation kind", maximum=80)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", kind):
            raise ValueError("observation kind must be an uppercase identifier")
        payload = observation.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError(f"observation {index} payload must be an object")
        payload_value = _snapshot(payload)
        _validate_pseudonymous_payload(payload_value)
        if len(_canonical_bytes(payload_value)) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"observation {index} payload is too large")
        normalized.append((observed, kind, payload_value))

    normalized.sort(key=lambda row: row[0])
    observations = []
    for index, (observed, kind, payload) in enumerate(normalized, start=1):
        delta = (cutoff - observed).total_seconds()
        if not math.isfinite(delta) or delta < 0 or not delta.is_integer():
            raise ValueError(
                "observation offsets must resolve to whole non-negative seconds"
            )
        observations.append(
            {
                "id": f"obs-{index:03d}",
                "seconds_before_cutoff": int(delta),
                "kind": kind,
                "payload": payload,
            }
        )

    # The private nonce salts the source commitment so a short list of market
    # dates cannot be dictionary-matched from the commitment alone. Exact price
    # paths remain re-identifiable, so the artifact is only pseudonymized.
    source_sha = canonical_sha256({**source_value, "blind_nonce": blind_nonce})
    trial_id = (
        "dojo-"
        + canonical_sha256(
            {
                "source_sha256": source_sha,
                "prompt_lock_sha256": prompt["prompt_lock_sha256"],
                "scorer_lock_sha256": scorer["scorer_lock_sha256"],
            }
        )[:24]
    )
    body = {
        "contract": PACKET_CONTRACT,
        "schema_version": 1,
        "validity_status": INVALIDATED if invalid_parents else VALID,
        "trial_id": trial_id,
        "pair": pair,
        "decision_time_label": "T0",
        "observations": observations,
        "source_sha256": source_sha,
        "prompt_lock_sha256": prompt["prompt_lock_sha256"],
        "capability_manifest_sha256": capability["capability_manifest_sha256"],
        "scorer_lock_sha256": scorer["scorer_lock_sha256"],
        "calendar_date_removed": True,
        "anonymity_status": "PSEUDONYMIZED_NOT_ANONYMOUS",
        "reidentification_risk": "EXACT_PRICE_PATH_MAY_REIDENTIFY_MARKET_DATE",
        "single_day_packet": True,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        **_diagnostic_fields(),
    }
    if invalid_parents:
        body["invalidation_reason"] = "INVALIDATED_PARENT:" + ",".join(invalid_parents)
    return _seal(body, "packet_sha256")


def seal_response(
    response: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    prompt_lock: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    capability_manifest: Mapping[str, Any],
    sealed_at_utc: datetime,
) -> dict[str, Any]:
    """Validate and seal one response without reading any answer material."""

    packet_value = _snapshot(packet)
    prompt = _snapshot(prompt_lock)
    model = _snapshot(model_manifest)
    capability = _snapshot(capability_manifest)
    _validate_day_packet(packet_value)
    _validate_prompt_lock(prompt)
    _validate_model_manifest(model)
    _validate_capability_manifest(capability)
    _validate_response_shape(response, packet_value)
    response_value = _snapshot(response)

    if packet_value["prompt_lock_sha256"] != prompt["prompt_lock_sha256"]:
        raise ValueError("packet prompt binding is stale")
    if (
        packet_value["capability_manifest_sha256"]
        != capability["capability_manifest_sha256"]
    ):
        raise ValueError("packet capability binding is stale")
    if model["capability_manifest_sha256"] != capability["capability_manifest_sha256"]:
        raise ValueError("model capability binding is stale")
    if model["context_id"] != capability["context_id"]:
        raise ValueError("model context binding is stale")

    sealed_at = _aware_utc(sealed_at_utc)
    prerequisite_times = (
        _parse_utc_required(prompt.get("locked_at_utc"), "prompt lock"),
        _parse_utc_required(model.get("locked_at_utc"), "model lock"),
        _parse_utc_required(capability.get("generated_at_utc"), "capability manifest"),
    )
    if any(instant > sealed_at for instant in prerequisite_times):
        raise ValueError(
            "prompt, model, and capability must be prelocked before response"
        )
    parent_values = (packet_value, prompt, model, capability)
    invalid_parents = _invalid_parent_contracts(parent_values)
    body = {
        "contract": RESPONSE_CONTRACT,
        "schema_version": 1,
        "validity_status": INVALIDATED if invalid_parents else VALID,
        "trial_id": packet_value["trial_id"],
        "sealed_at_utc": sealed_at.isoformat(),
        "packet_sha256": packet_value["packet_sha256"],
        "prompt_lock_sha256": prompt["prompt_lock_sha256"],
        "prompt_sha256": prompt["prompt_sha256"],
        "model_sha256": model["model_sha256"],
        "declared_model_lineage": model["declared_model_lineage"],
        "model_lineage_attestation_status": model["model_lineage_attestation_status"],
        "capability_manifest_sha256": capability["capability_manifest_sha256"],
        "scorer_lock_sha256": packet_value["scorer_lock_sha256"],
        "response": response_value,
        "response_sha256": canonical_sha256(response_value),
        "answer_key_opened": False,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "parent_bindings_revalidated": True,
        "chronology_attestation_status": "SELF_ATTESTED_UNVERIFIED",
        **_diagnostic_fields(),
    }
    if invalid_parents:
        body["invalidation_reason"] = "INVALIDATED_PARENT:" + ",".join(invalid_parents)
    return _seal(body, "response_receipt_sha256")


def seal_answer_key(
    *,
    trial_id: str,
    packet_sha256: str,
    returns: Mapping[str, Any],
    sealed_at_utc: datetime,
) -> dict[str, Any]:
    """Seal offline truth; this artifact must never be mounted in model context."""

    trial = _bounded_text(trial_id, "trial_id", maximum=100)
    packet_sha = _require_sha256(packet_sha256, "packet_sha256")
    if not isinstance(returns, Mapping) or set(returns) != _RETURN_KEYS:
        raise ValueError("answer returns shape is invalid")
    normalized_returns: dict[str, float] = {}
    for key in sorted(_RETURN_KEYS):
        value = _finite_number(returns[key], f"return {key}")
        if value <= -1.0:
            raise ValueError("returns must keep the account multiplier positive")
        normalized_returns[key] = value
    if normalized_returns["FLAT"] != 0.0:
        raise ValueError("FLAT return must be exactly zero")
    body = {
        "contract": ANSWER_KEY_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "trial_id": trial,
        "packet_sha256": packet_sha,
        "sealed_at_utc": _aware_utc(sealed_at_utc).isoformat(),
        "returns": normalized_returns,
        "answer_key_provenance_status": "SELF_ATTESTED_UNVERIFIED",
        "market_data_attestation_sha256": None,
        "cost_model_attestation_sha256": None,
        "chronology_attestation_status": "SELF_ATTESTED_UNVERIFIED",
        "read_only": True,
        "live_permission": False,
        **_diagnostic_fields(),
    }
    return _seal(body, "answer_key_sha256")


def score_sealed_response(
    response_receipt: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    prompt_lock: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    capability_manifest: Mapping[str, Any],
    scorer_lock: Mapping[str, Any],
    answer_key_loader: Callable[[], Mapping[str, Any]],
    opened_at_utc: datetime,
) -> dict[str, Any]:
    """Verify response/scorer seals before invoking the answer-key loader."""

    response = _snapshot(response_receipt)
    scorer = _snapshot(scorer_lock)
    packet_value = _snapshot(packet)
    prompt = _snapshot(prompt_lock)
    model = _snapshot(model_manifest)
    capability = _snapshot(capability_manifest)
    _validate_day_packet(packet_value)
    _validate_response_receipt(response, packet=packet_value)
    _validate_scorer_lock(scorer)
    _validate_prompt_lock(prompt)
    _validate_model_manifest(model)
    _validate_capability_manifest(capability)
    if response.get("prompt_lock_sha256") != prompt.get("prompt_lock_sha256"):
        raise ValueError("response prompt parent binding is stale")
    if response.get("model_sha256") != model.get("model_sha256"):
        raise ValueError("response model parent binding is stale")
    if response.get("capability_manifest_sha256") != capability.get(
        "capability_manifest_sha256"
    ):
        raise ValueError("response capability parent binding is stale")
    if model.get("capability_manifest_sha256") != capability.get(
        "capability_manifest_sha256"
    ):
        raise ValueError("model capability parent binding is stale")
    bound_scorer_sha = response.get("scorer_lock_sha256")
    scorer_matches = bound_scorer_sha == scorer.get("scorer_lock_sha256")
    scorer_invalidates_bound = (
        scorer.get("validity_status") == INVALIDATED
        and scorer.get("invalidates_artifact_sha256") == bound_scorer_sha
    )
    if not scorer_matches and not scorer_invalidates_bound:
        raise ValueError("response scorer binding is stale")
    response_sealed = _parse_utc_required(
        response.get("sealed_at_utc"), "response seal"
    )
    scorer_locked = _parse_utc_required(scorer.get("locked_at_utc"), "scorer lock")
    if scorer_locked > response_sealed:
        raise ValueError("scorer must be prelocked before response")
    if (
        response["validity_status"] == INVALIDATED
        or scorer["validity_status"] == INVALIDATED
    ):
        return _invalidated_score(
            response, scorer, "INVALIDATED_PARENT_BEFORE_KEY_OPEN"
        )
    if response["validity_status"] != VALID or scorer["validity_status"] != VALID:
        raise ValueError("response or scorer validity is unsupported")

    opened = _aware_utc(opened_at_utc)
    if opened <= response_sealed:
        raise ValueError("answer key may open only after response seal")

    loaded = answer_key_loader()
    if not isinstance(loaded, Mapping):
        raise ValueError("answer key loader must return one JSON object")
    answer_key = _snapshot(loaded)
    _validate_answer_key(answer_key)
    if answer_key["validity_status"] == INVALIDATED:
        return _invalidated_score(response, scorer, "ANSWER_KEY_INVALIDATED")
    if answer_key["validity_status"] != VALID:
        raise ValueError("answer key validity is unsupported")
    if answer_key["trial_id"] != response["trial_id"]:
        raise ValueError("answer key trial binding is stale")
    if answer_key["packet_sha256"] != response["packet_sha256"]:
        raise ValueError("answer key packet binding is stale")
    answer_key_sealed = _parse_utc_required(
        answer_key.get("sealed_at_utc"), "answer key seal"
    )
    if answer_key_sealed <= response_sealed:
        raise ValueError("answer key must be sealed after response seal")
    if answer_key_sealed > opened:
        raise ValueError("answer key cannot be sealed after it is opened")

    decision = response["response"]
    action = decision["action"]
    size = decision["size"]
    return_key = "FLAT" if action == "FLAT" else f"{action}_{size}"
    net_return = _finite_number(answer_key["returns"][return_key], "selected return")
    log_growth = math.log1p(net_return)
    score_core = {
        "trial_id": response["trial_id"],
        "return_key": return_key,
        "net_return": net_return,
        "net_log_growth": log_growth,
        "response_sha256": response["response_sha256"],
        "answer_key_sha256": answer_key["answer_key_sha256"],
        "scorer_sha256": scorer["scorer_sha256"],
    }
    body = {
        "contract": SCORE_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "trial_id": response["trial_id"],
        "scored_at_utc": opened.isoformat(),
        "declared_model_lineage": response["declared_model_lineage"],
        "model_lineage_attestation_status": response[
            "model_lineage_attestation_status"
        ],
        "packet_sha256": response["packet_sha256"],
        "prompt_lock_sha256": response["prompt_lock_sha256"],
        "prompt_sha256": response["prompt_sha256"],
        "model_sha256": response["model_sha256"],
        "response_sha256": response["response_sha256"],
        "response_receipt_sha256": response["response_receipt_sha256"],
        "scorer_lock_sha256": scorer["scorer_lock_sha256"],
        "scorer_sha256": scorer["scorer_sha256"],
        "answer_key_sha256": answer_key["answer_key_sha256"],
        "answer_key_opened_after_response_seal": True,
        "answer_key_sealed_after_response_seal": True,
        "return_key": return_key,
        "net_return": net_return,
        "net_log_growth": log_growth,
        "score_sha256": canonical_sha256(score_core),
        "diagnostic_return_sign": "POSITIVE"
        if net_return > 0.0
        else "NEGATIVE"
        if net_return < 0.0
        else "ZERO",
        "diagnostic_status": "DIAGNOSTIC_ONLY",
        "parent_bindings_revalidated_in_process": True,
        "answer_key_provenance_status": answer_key["answer_key_provenance_status"],
        "proof_eligible": False,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        **_diagnostic_fields(),
    }
    return _seal(body, "score_receipt_sha256")


def score_pilot(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate sealed trial scores without treating one lineage as iid agents."""

    if (
        isinstance(scores, (str, bytes))
        or not isinstance(scores, Sequence)
        or not scores
    ):
        raise ValueError("pilot requires a non-empty score sequence")
    snapshots = [_snapshot(score) for score in scores]
    for score in snapshots:
        _validate_score_receipt(score)
    trial_ids = [str(score.get("trial_id") or "") for score in snapshots]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("pilot contains duplicate trial ids")

    invalid = [score for score in snapshots if score["validity_status"] == INVALIDATED]
    lineage_counts: dict[str, int] = {}
    for score in snapshots:
        lineage = str(score.get("declared_model_lineage") or "UNKNOWN")
        lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1

    if invalid:
        body = {
            "contract": PILOT_SCORE_CONTRACT,
            "schema_version": 1,
            "validity_status": INVALIDATED,
            "invalidation_reason": "TRANSITIVE_INVALIDATED_TRIAL_SCORE",
            "trial_count": len(snapshots),
            "declared_lineage_cluster_count_diagnostic": len(lineage_counts),
            "declared_lineage_trial_counts": dict(sorted(lineage_counts.items())),
            "model_lineage_attestation_status": "DECLARED_ONLY_NOT_PROVIDER_ATTESTED",
            "score_receipt_sha256s": [
                score["score_receipt_sha256"] for score in snapshots
            ],
            "compounded_net_return": None,
            "total_log_growth": None,
            "diagnostic_status": "INVALIDATED",
            "proof_eligible": False,
            "read_only": True,
            "ai_order_authority": "NONE",
            "live_permission": False,
            "broker_mutation_allowed": False,
            **_diagnostic_fields(),
        }
        return _seal(body, "pilot_score_sha256")

    total_log = sum(
        _finite_number(score["net_log_growth"], "score net log growth")
        for score in snapshots
    )
    compounded_return = math.expm1(total_log)
    body = {
        "contract": PILOT_SCORE_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "trial_count": len(snapshots),
        "declared_lineage_cluster_count_diagnostic": len(lineage_counts),
        "declared_lineage_trial_counts": dict(sorted(lineage_counts.items())),
        "model_lineage_attestation_status": "DECLARED_ONLY_NOT_PROVIDER_ATTESTED",
        "score_receipt_sha256s": [score["score_receipt_sha256"] for score in snapshots],
        "compounded_net_return": compounded_return,
        "total_log_growth": total_log,
        "diagnostic_return_sign": "POSITIVE"
        if total_log > 0.0
        else "NEGATIVE"
        if total_log < 0.0
        else "ZERO",
        "diagnostic_status": "DIAGNOSTIC_ONLY",
        "proof_eligible": False,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        **_diagnostic_fields(),
    }
    return _seal(body, "pilot_score_sha256")


def validate_score_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a finite snapshot after validating one diagnostic score seal."""

    value = _snapshot(receipt)
    _validate_score_receipt(value)
    return value


def invalidate_artifact(
    artifact: Mapping[str, Any],
    *,
    reason: str,
    evidence_sha256: str,
    invalidated_at_utc: datetime,
) -> dict[str, Any]:
    """Create a new sealed invalidated revision without erasing old bytes."""

    value = _snapshot(artifact)
    own_field = _validate_known_seal(value)
    invalidated_sha = value.get("invalidates_artifact_sha256") or value[own_field]
    body = {key: item for key, item in value.items() if key != own_field}
    body["validity_status"] = INVALIDATED
    body["invalidates_artifact_sha256"] = _require_sha256(
        invalidated_sha, "invalidated artifact sha256"
    )
    body["invalidation_reason"] = _bounded_text(
        reason, "invalidation reason", maximum=500
    )
    body["invalidation_evidence_sha256"] = _require_sha256(
        evidence_sha256, "invalidation evidence sha256"
    )
    body["invalidated_at_utc"] = _aware_utc(invalidated_at_utc).isoformat()
    body.pop("positive_evidence", None)
    body.pop("effective_independent_n", None)
    body.pop("verdict", None)
    body["diagnostic_status"] = "INVALIDATED"
    if value.get("contract") == SCORE_CONTRACT:
        body["return_key"] = None
        body["net_return"] = None
        body["net_log_growth"] = None
        body["score_sha256"] = None
    elif value.get("contract") == PILOT_SCORE_CONTRACT:
        body["compounded_net_return"] = None
        body["total_log_growth"] = None
    return _seal(body, own_field)


def seal_validity_registry(
    heads: Mapping[str, str], *, updated_at_utc: datetime
) -> dict[str, Any]:
    """Seal the current status of exact artifact bytes for stale-copy rejection."""

    if not isinstance(heads, Mapping) or not heads:
        raise ValueError("validity registry requires at least one head")
    normalized: dict[str, str] = {}
    for artifact_sha, status in sorted(heads.items()):
        digest = _require_sha256(artifact_sha, "validity head artifact sha256")
        if status not in {VALID, INVALIDATED}:
            raise ValueError("validity head status must be VALID or INVALIDATED")
        normalized[digest] = status
    body = {
        "contract": VALIDITY_REGISTRY_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "updated_at_utc": _aware_utc(updated_at_utc).isoformat(),
        "heads": normalized,
        "read_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "monotonicity_status": "NOT_EXTERNALLY_MONOTONIC",
        **_diagnostic_fields(),
    }
    return _seal(body, "validity_registry_sha256")


def assert_no_stale_positive_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    validity_registry: Mapping[str, Any] | None = None,
) -> None:
    """Reject legacy/invalid/tampered artifacts that still advertise a win."""

    registry = _snapshot(validity_registry) if validity_registry is not None else None
    if registry is not None:
        _validate_validity_registry(registry)
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("stale positive artifact: artifact is not an object")
        value = _snapshot(artifact)
        if not _claims_positive(value):
            continue
        if value.get("validity_status") != VALID:
            raise ValueError(
                "stale positive artifact: validity is missing or not VALID"
            )
        try:
            own_field = _validate_known_seal(value)
        except ValueError as exc:
            raise ValueError(
                "stale positive artifact: missing or invalid active seal"
            ) from exc
        if registry is None:
            raise ValueError(
                "stale positive artifact: current validity head is required"
            )
        artifact_sha = str(value[own_field])
        if registry["heads"].get(artifact_sha) != VALID:
            raise ValueError("stale positive artifact: validity head is not VALID")


def _invalidated_score(
    response: Mapping[str, Any], scorer: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    body = {
        "contract": SCORE_CONTRACT,
        "schema_version": 1,
        "validity_status": INVALIDATED,
        "invalidation_reason": reason,
        "trial_id": response.get("trial_id"),
        "scored_at_utc": None,
        "declared_model_lineage": response.get("declared_model_lineage"),
        "model_lineage_attestation_status": response.get(
            "model_lineage_attestation_status"
        ),
        "packet_sha256": response.get("packet_sha256"),
        "prompt_lock_sha256": response.get("prompt_lock_sha256"),
        "prompt_sha256": response.get("prompt_sha256"),
        "model_sha256": response.get("model_sha256"),
        "response_sha256": response.get("response_sha256"),
        "response_receipt_sha256": response.get("response_receipt_sha256"),
        "scorer_lock_sha256": scorer.get("scorer_lock_sha256"),
        "scorer_sha256": scorer.get("scorer_sha256"),
        "answer_key_sha256": None,
        "answer_key_opened_after_response_seal": False,
        "return_key": None,
        "net_return": None,
        "net_log_growth": None,
        "score_sha256": None,
        "diagnostic_status": "INVALIDATED",
        "proof_eligible": False,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        **_diagnostic_fields(),
    }
    return _seal(body, "score_receipt_sha256")


def _validate_pseudonymous_payload(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            normalized = key.strip().lower()
            if (
                normalized in _DATE_PAYLOAD_KEYS
                or normalized.endswith("_date")
                or normalized.endswith("_utc")
            ):
                raise ValueError(
                    f"{path} contains forbidden date/timestamp field: {key}"
                )
            if any(part in normalized for part in _FORBIDDEN_PAYLOAD_KEY_PARTS):
                raise ValueError(f"{path} contains forward/outcome field: {key}")
            _validate_pseudonymous_payload(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_pseudonymous_payload(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if _DATE_TEXT.search(value):
            raise ValueError(f"{path} contains identifiable calendar date")
        if _looks_like_temporal_encoding(value):
            raise ValueError(f"{path} contains obvious numeric date/epoch encoding")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if _looks_like_temporal_encoding(value):
            raise ValueError(f"{path} contains obvious numeric date/epoch encoding")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite number")


def _looks_like_temporal_encoding(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    text: str | None = None
    if isinstance(value, int):
        text = str(abs(value))
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return False
        text = str(abs(int(value)))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            text = stripped
        else:
            return any(
                _looks_like_temporal_encoding(token)
                for token in re.findall(r"(?<!\d)\d{8,16}(?!\d)", stripped)
            )
    if text is None:
        return False

    integer = int(text)
    if 946_684_800 <= integer <= 4_102_444_800:
        return True
    if 946_684_800_000 <= integer <= 4_102_444_800_000:
        return True
    if 946_684_800_000_000 <= integer <= 4_102_444_800_000_000:
        return True
    formats = {8: "%Y%m%d", 12: "%Y%m%d%H%M", 14: "%Y%m%d%H%M%S"}
    format_text = formats.get(len(text))
    if format_text is None:
        return False
    try:
        parsed = datetime.strptime(text, format_text)
    except ValueError:
        return False
    return 2000 <= parsed.year <= 2100


def _validate_response_shape(
    response: Mapping[str, Any], packet: Mapping[str, Any]
) -> None:
    if not isinstance(response, Mapping) or set(response) != FIXED_RESPONSE_KEYS:
        raise ValueError("fixed response schema mismatch")
    value = _snapshot(response)
    if value.get("trial_id") != packet.get("trial_id"):
        raise ValueError("response trial_id does not match packet")
    if value.get("pair") != packet.get("pair"):
        raise ValueError("response pair does not match packet")
    action = value.get("action")
    size = value.get("size")
    if action not in {"FLAT", "LONG", "SHORT"}:
        raise ValueError("fixed response schema action is invalid")
    if size not in {"NONE", "HALF", "FULL"}:
        raise ValueError("fixed response schema size is invalid")
    confidence = _finite_number(value.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be within 0..1")
    references = value.get("evidence_refs")
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(item, str) for item in references)
        or len(set(references)) != len(references)
    ):
        raise ValueError("evidence_refs must be a non-empty unique string list")
    allowed_refs = {row["id"] for row in packet.get("observations", [])}
    if not set(references).issubset(allowed_refs):
        raise ValueError("response cites evidence outside its packet")
    counterargument = str(value.get("strongest_counterargument") or "").strip()
    if not 1 <= len(counterargument) <= 1_000:
        raise ValueError("strongest_counterargument is required and bounded")
    if action == "FLAT":
        if (
            size != "NONE"
            or value.get("target_pips") is not None
            or value.get("invalidation_pips") is not None
        ):
            raise ValueError("FLAT response must use NONE size and null geometry")
        _bounded_text(value.get("abstain_reason"), "abstain_reason", maximum=1_000)
    else:
        if size not in {"HALF", "FULL"} or value.get("abstain_reason") is not None:
            raise ValueError("directional response size/abstain fields are invalid")
        if _finite_number(value.get("target_pips"), "target_pips") <= 0:
            raise ValueError("target_pips must be positive")
        if _finite_number(value.get("invalidation_pips"), "invalidation_pips") <= 0:
            raise ValueError("invalidation_pips must be positive")


def _validate_capability_manifest(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, CAPABILITY_MANIFEST_CONTRACT)
    allowed_keys = _CAPABILITY_KEYS | _INVALIDATION_METADATA_KEYS
    if not _CAPABILITY_KEYS.issubset(value) or not set(value).issubset(allowed_keys):
        raise ValueError("capability manifest shape is invalid")
    if value.get("validity_status") == VALID and set(value) != _CAPABILITY_KEYS:
        raise ValueError("valid capability manifest contains invalidation metadata")
    if value.get("validity_status") == INVALIDATED and not isinstance(
        value.get("invalidation_reason"), str
    ):
        raise ValueError("invalidated capability manifest lacks an invalidation reason")
    required_false = (
        "declared_answer_key_access",
        "declared_repository_access",
        "declared_filesystem_access",
        "declared_network_access",
        "declared_browser_access",
        "declared_conversation_history_access",
        "declared_persistent_memory_access",
        "external_send_allowed",
        "model_api_invoked_by_runtime",
        "live_permission",
        "broker_mutation_allowed",
    )
    if value.get("declared_fresh_context") is not True:
        raise ValueError("capability manifest is not a fresh context")
    if value.get("declared_answer_key_physically_absent") is not True:
        raise ValueError("capability manifest does not exclude the answer key")
    if (
        value.get("declared_mounted_artifact_roles") != ["INLINE_PACKET_ONLY"]
        or value.get("declared_tools") != []
    ):
        raise ValueError("capability manifest exposes unsupported capabilities")
    if any(value.get(key) is not False for key in required_false):
        raise ValueError("capability manifest exposes unsupported capabilities")


def _validate_prompt_lock(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, PROMPT_LOCK_CONTRACT)
    allowed = _PROMPT_LOCK_KEYS | _INVALIDATION_METADATA_KEYS
    if not _PROMPT_LOCK_KEYS.issubset(value) or not set(value).issubset(allowed):
        raise ValueError("prompt lock shape is invalid")
    if value.get("validity_status") == VALID and set(value) != _PROMPT_LOCK_KEYS:
        raise ValueError("valid prompt lock contains invalidation metadata")
    if value.get("fixed_output_schema") != FIXED_RESPONSE_SCHEMA:
        raise ValueError("prompt fixed output schema is stale")
    if value.get("output_schema_sha256") != canonical_sha256(FIXED_RESPONSE_SCHEMA):
        raise ValueError("prompt output schema digest is stale")
    prompt_text = value.get("prompt_text")
    if (
        not isinstance(prompt_text, str)
        or value.get("prompt_sha256")
        != hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    ):
        raise ValueError("prompt digest is stale")


def _validate_scorer_lock(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, SCORER_LOCK_CONTRACT)
    allowed = _SCORER_LOCK_KEYS | _INVALIDATION_METADATA_KEYS
    if not _SCORER_LOCK_KEYS.issubset(value) or not set(value).issubset(allowed):
        raise ValueError("scorer lock shape is invalid")
    if value.get("validity_status") == VALID and set(value) != _SCORER_LOCK_KEYS:
        raise ValueError("valid scorer lock contains invalidation metadata")
    if value.get("policy") != _SCORER_POLICY or value.get(
        "scorer_sha256"
    ) != canonical_sha256(_SCORER_POLICY):
        raise ValueError("scorer policy is stale")


def _validate_model_manifest(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, MODEL_MANIFEST_CONTRACT)
    if (
        value.get("declared_fresh_context") is not True
        or value.get("model_api_invoked_by_runtime") is not False
        or value.get("external_send_allowed") is not False
        or value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
    ):
        raise ValueError("model manifest violates offline-only boundary")
    _bounded_text(
        value.get("declared_model_lineage"), "declared_model_lineage", maximum=200
    )
    if value.get("model_lineage_attestation_status") != (
        "DECLARED_ONLY_NOT_PROVIDER_ATTESTED"
    ):
        raise ValueError("model lineage is not honestly labeled as declared only")
    _require_sha256(
        value.get("capability_manifest_sha256"), "capability manifest sha256"
    )


def _validate_day_packet(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, PACKET_CONTRACT)
    allowed_keys = _PACKET_KEYS | _INVALIDATION_METADATA_KEYS
    if not _PACKET_KEYS.issubset(value) or not set(value).issubset(allowed_keys):
        raise ValueError("packet shape is invalid")
    if value.get("validity_status") == VALID and set(value) != _PACKET_KEYS:
        raise ValueError("valid packet contains invalidation metadata")
    if value.get("validity_status") == INVALIDATED and not isinstance(
        value.get("invalidation_reason"), str
    ):
        raise ValueError("invalidated packet lacks an invalidation reason")
    pair = value.get("pair")
    if not isinstance(pair, str) or not re.fullmatch(r"[A-Z]{3}_[A-Z]{3}", pair):
        raise ValueError("packet pair is invalid")
    trial_id = value.get("trial_id")
    if not isinstance(trial_id, str) or not re.fullmatch(
        r"dojo-[0-9a-f]{24}", trial_id
    ):
        raise ValueError("packet trial id is invalid")
    for key in (
        "source_sha256",
        "prompt_lock_sha256",
        "capability_manifest_sha256",
        "scorer_lock_sha256",
    ):
        _require_sha256(value.get(key), key)
    observations = value.get("observations")
    if (
        not isinstance(observations, list)
        or not 1 <= len(observations) <= MAX_OBSERVATIONS
    ):
        raise ValueError("packet observations are invalid")
    prior_offset: int | None = None
    for index, raw in enumerate(observations, start=1):
        if not isinstance(raw, Mapping) or set(raw) != _PACKET_OBSERVATION_KEYS:
            raise ValueError("packet observation shape is invalid")
        if raw.get("id") != f"obs-{index:03d}":
            raise ValueError("packet observation ids are not canonical")
        offset = raw.get("seconds_before_cutoff")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("packet observation offset is invalid")
        if prior_offset is not None and offset > prior_offset:
            raise ValueError("packet observations are not chronologically ordered")
        prior_offset = offset
        kind = raw.get("kind")
        if not isinstance(kind, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", kind):
            raise ValueError("packet observation kind is invalid")
        payload = raw.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("packet observation payload is invalid")
        _validate_pseudonymous_payload(payload)
        if len(_canonical_bytes(payload)) > MAX_PAYLOAD_BYTES:
            raise ValueError("packet observation payload is too large")
    model_visible = {
        "trial_id": value.get("trial_id"),
        "pair": value.get("pair"),
        "decision_time_label": value.get("decision_time_label"),
        "observations": value.get("observations"),
    }
    if (
        value.get("single_day_packet") is not True
        or value.get("calendar_date_removed") is not True
        or value.get("anonymity_status") != "PSEUDONYMIZED_NOT_ANONYMOUS"
        or value.get("reidentification_risk")
        != "EXACT_PRICE_PATH_MAY_REIDENTIFY_MARKET_DATE"
        or value.get("decision_time_label") != "T0"
        or _DATE_TEXT.search(json.dumps(model_visible, ensure_ascii=False))
    ):
        raise ValueError("packet is not one calendar-date-removed pseudonymous day")
    if (
        value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
    ):
        raise ValueError("packet violates offline-only boundary")


def _validate_response_receipt(
    value: Mapping[str, Any], *, packet: Mapping[str, Any] | None = None
) -> None:
    try:
        _validate_contract_seal(value, RESPONSE_CONTRACT)
    except ValueError as exc:
        raise ValueError("response receipt seal is invalid") from exc
    if value.get("validity_status") == VALID:
        response = value.get("response")
        if not isinstance(response, Mapping) or canonical_sha256(response) != value.get(
            "response_sha256"
        ):
            raise ValueError("response receipt seal is invalid")
        if packet is not None:
            if value.get("packet_sha256") != packet.get("packet_sha256"):
                raise ValueError("response packet parent binding is stale")
            _validate_response_shape(response, packet)
    if value.get("answer_key_opened") is not False:
        raise ValueError("response receipt claims premature answer-key access")
    if value.get("model_lineage_attestation_status") != (
        "DECLARED_ONLY_NOT_PROVIDER_ATTESTED"
    ):
        raise ValueError("response lineage is not honestly labeled as declared only")
    if (
        value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
    ):
        raise ValueError("response receipt violates offline-only boundary")


def _validate_answer_key(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, ANSWER_KEY_CONTRACT)
    returns = value.get("returns")
    if not isinstance(returns, Mapping) or set(returns) != _RETURN_KEYS:
        raise ValueError("answer key returns shape is invalid")
    for key in _RETURN_KEYS:
        number = _finite_number(returns[key], f"answer return {key}")
        if number <= -1.0:
            raise ValueError("answer key return is invalid")
    if (
        value.get("answer_key_provenance_status") != "SELF_ATTESTED_UNVERIFIED"
        or value.get("market_data_attestation_sha256") is not None
        or value.get("cost_model_attestation_sha256") is not None
    ):
        raise ValueError("answer key overstates trusted market provenance")


def _validate_score_receipt(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, SCORE_CONTRACT)
    if (
        value.get("live_permission") is not False
        or value.get("proof_eligible") is not False
    ):
        raise ValueError("score receipt violates audit-only boundary")
    if value.get("validity_status") == INVALIDATED:
        if (
            value.get("net_return") is not None
            or value.get("net_log_growth") is not None
        ):
            raise ValueError("invalidated score retains positive metrics")
        if value.get("diagnostic_status") != "INVALIDATED":
            raise ValueError("invalidated score retains an active diagnostic status")
    elif value.get("validity_status") != VALID:
        raise ValueError("score receipt validity is unsupported")
    else:
        net_return = _finite_number(value.get("net_return"), "score net return")
        net_log = _finite_number(value.get("net_log_growth"), "score net log growth")
        if net_return <= -1.0 or not math.isclose(
            net_log, math.log1p(net_return), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("score math does not match net return")
        score_core = {
            "trial_id": value.get("trial_id"),
            "return_key": value.get("return_key"),
            "net_return": net_return,
            "net_log_growth": net_log,
            "response_sha256": value.get("response_sha256"),
            "answer_key_sha256": value.get("answer_key_sha256"),
            "scorer_sha256": value.get("scorer_sha256"),
        }
        if value.get("score_sha256") != canonical_sha256(score_core):
            raise ValueError("score math receipt digest is stale")
        for key in (
            "packet_sha256",
            "prompt_lock_sha256",
            "prompt_sha256",
            "model_sha256",
            "response_sha256",
            "response_receipt_sha256",
            "scorer_lock_sha256",
            "scorer_sha256",
            "answer_key_sha256",
        ):
            _require_sha256(value.get(key), key)
        _bounded_text(
            value.get("declared_model_lineage"),
            "declared_model_lineage",
            maximum=200,
        )
        if value.get("model_lineage_attestation_status") != (
            "DECLARED_ONLY_NOT_PROVIDER_ATTESTED"
        ):
            raise ValueError("score lineage is not honestly labeled as declared only")
        expected_sign = (
            "POSITIVE"
            if net_return > 0.0
            else "NEGATIVE"
            if net_return < 0.0
            else "ZERO"
        )
        if value.get("diagnostic_return_sign") != expected_sign:
            raise ValueError("score diagnostic sign does not match net return")
        if value.get("diagnostic_status") != "DIAGNOSTIC_ONLY":
            raise ValueError("score diagnostic status is invalid")


def _validate_validity_registry(value: Mapping[str, Any]) -> None:
    _validate_contract_seal(value, VALIDITY_REGISTRY_CONTRACT)
    if value.get("validity_status") != VALID:
        raise ValueError("validity registry itself is not VALID")
    if value.get("monotonicity_status") != "NOT_EXTERNALLY_MONOTONIC":
        raise ValueError("validity registry overstates monotonicity")
    heads = value.get("heads")
    if not isinstance(heads, Mapping) or not heads:
        raise ValueError("validity registry heads are missing")
    for artifact_sha, status in heads.items():
        _require_sha256(artifact_sha, "validity head artifact sha256")
        if status not in {VALID, INVALIDATED}:
            raise ValueError("validity head status is unsupported")


def _validate_contract_seal(value: Mapping[str, Any], contract: str) -> None:
    if value.get("contract") != contract:
        raise ValueError(f"expected {contract} artifact")
    own_field = _OWN_SHA_FIELDS[contract]
    claimed = value.get(own_field)
    body = {key: item for key, item in value.items() if key != own_field}
    if not _is_sha256(claimed) or claimed != canonical_sha256(body):
        raise ValueError(f"invalid {contract} seal")
    if value.get("validity_status") not in {VALID, INVALIDATED}:
        raise ValueError(f"invalid {contract} validity")
    _validate_diagnostic_boundary(value)


def _validate_diagnostic_boundary(value: Mapping[str, Any]) -> None:
    if (
        value.get("evidence_tier") != DIAGNOSTIC_TIER
        or value.get("external_attestations_verified") is not False
        or value.get("attestation_gap_codes") != _ATTESTATION_GAPS
    ):
        raise ValueError("artifact overstates external attestation")
    if value.get("live_permission") is not False:
        raise ValueError("diagnostic artifact grants live permission")
    if value.get("broker_mutation_allowed") not in {None, False}:
        raise ValueError("diagnostic artifact grants broker mutation")
    if value.get("ai_order_authority") not in {None, "NONE"}:
        raise ValueError("diagnostic artifact grants AI order authority")
    forbidden_fields = {
        "positive_evidence",
        "effective_independent_n",
        "promotion_status",
        "promotion_eligible",
        "automatic_promotion_allowed",
        "verdict",
    }
    if forbidden_fields.intersection(value):
        raise ValueError("diagnostic artifact contains proof or promotion semantics")
    if value.get("proof_eligible") not in {None, False}:
        raise ValueError("diagnostic artifact contains proof or promotion semantics")
    serialized = json.dumps(value, ensure_ascii=False).upper()
    if "EDGE_PROVEN" in serialized or "LIVE_ELIGIBLE" in serialized:
        raise ValueError("diagnostic artifact contains forbidden promotion vocabulary")


def _validate_known_seal(value: Mapping[str, Any]) -> str:
    contract = value.get("contract")
    if contract not in _OWN_SHA_FIELDS:
        raise ValueError("artifact contract has no registered active seal")
    _validate_contract_seal(value, str(contract))
    return _OWN_SHA_FIELDS[str(contract)]


def _invalid_parent_contracts(values: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(value.get("contract"))
        for value in values
        if value.get("validity_status") == INVALIDATED
    ]


def _claims_positive(value: Mapping[str, Any]) -> bool:
    if value.get("positive_evidence") is True:
        return True
    if str(value.get("diagnostic_return_sign") or "").upper() == "POSITIVE":
        return True
    for key in (
        "net_return",
        "net_log_growth",
        "compounded_net_return",
        "total_log_growth",
    ):
        raw = value.get(key)
        if (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
            and float(raw) > 0.0
        ):
            return True
    verdict = str(value.get("verdict") or "").upper()
    return verdict.startswith("FIRST_POSITIVE") or verdict in {
        "POSITIVE",
        "POSITIVE_HYPOTHESIS_ONLY",
        "PASS",
        "BEATS_BASELINE",
        "PROMOTION_READY",
    }


def _seal(body: Mapping[str, Any], sha_field: str) -> dict[str, Any]:
    snapshot = _snapshot(body)
    if sha_field in snapshot:
        snapshot.pop(sha_field)
    return {**snapshot, sha_field: canonical_sha256(snapshot)}


def _snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact must be finite JSON data") from exc
    if not isinstance(decoded, dict):
        raise ValueError("artifact must be one JSON object")
    return decoded


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _bounded_text(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not 1 <= len(text) <= maximum:
        raise ValueError(f"{field} is required and bounded")
    return text


def _bounded_exact_text(value: Any, field: str, *, maximum: int) -> str:
    """Validate text while preserving every registered byte-equivalent char."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} is required and bounded")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _require_sha256(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase sha256")
    return str(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_utc_required(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an aware UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an aware UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be an aware UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "ANSWER_KEY_CONTRACT",
    "CAPABILITY_MANIFEST_CONTRACT",
    "FIXED_RESPONSE_KEYS",
    "FIXED_RESPONSE_SCHEMA",
    "MODEL_MANIFEST_CONTRACT",
    "PACKET_CONTRACT",
    "PILOT_SCORE_CONTRACT",
    "PROMPT_LOCK_CONTRACT",
    "RESPONSE_CONTRACT",
    "SCORER_LOCK_CONTRACT",
    "SCORE_CONTRACT",
    "SOURCE_CONTRACT",
    "VALIDITY_REGISTRY_CONTRACT",
    "assert_no_stale_positive_artifacts",
    "build_capability_manifest",
    "build_day_packet",
    "canonical_sha256",
    "invalidate_artifact",
    "prelock_prompt",
    "prelock_scorer",
    "score_pilot",
    "score_sealed_response",
    "seal_answer_key",
    "seal_model_manifest",
    "seal_response",
    "seal_validity_registry",
    "validate_score_receipt",
]
