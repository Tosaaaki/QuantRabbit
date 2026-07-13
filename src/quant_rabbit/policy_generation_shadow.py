"""Read-only, generation-scoped exact-vehicle evidence diagnostics.

The production all-exit gate deliberately aggregates broker-truth outcomes by
``pair / side / method / vehicle`` across the complete audited ledger.  That
surface has no policy-generation dimension.  This module provides a separate
shadow calculation for a future, explicitly tagged generation without changing
or replacing the authoritative all-time gate.

The evaluator is intentionally pure: it performs no file I/O, does not import
the live gateway, and always reports that it cannot grant live permission or
relax a gate.  A post-activation entry is usable only when its intent was
generated strictly after a content-addressed activation and precedes both the
gateway send and broker entry.  The entry must also bind the same generation
and activation digest.  Every observation carries the SHA-256 of the exact
immutable intent artifact bytes.  This evaluator validates only the digest's
shape; an external ledger adapter must verify the actual bytes.  Resolved rows
likewise bind an external content-addressed reconciliation of partial
reductions, terminal close, and financing.  Missing bindings fail the shadow
closed instead of being inferred from a timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


POLICY_GENERATION_SHADOW_CONTRACT = "QR_POLICY_GENERATION_EXACT_VEHICLE_SHADOW_V1"
_ACTIVATION_BODY_KEYS = {
    "contract",
    "policy_generation_id",
    "exact_lane_id",
    "activated_at_utc",
    "deployment_artifact_sha256",
    "source_revision",
    "read_only",
    "live_permission_allowed",
}
_ACTIVATION_KEYS = _ACTIVATION_BODY_KEYS | {"activation_sha256"}
_GENERATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_LANE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]{1,96}$")
_EVENT_UID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_RFC3339_UTC_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<clock>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?:Z|\+00:00)$"
)


@dataclass(frozen=True, order=True)
class _UtcInstant:
    """UTC instant whose ordering preserves OANDA's nanosecond precision."""

    epoch_second: int
    nanosecond: int

    def canonical(self) -> str:
        second = datetime.fromtimestamp(self.epoch_second, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        fraction = f".{self.nanosecond:09d}" if self.nanosecond else ""
        return f"{second}{fraction}+00:00"


@dataclass(frozen=True)
class GenerationScopedOutcomeObservation:
    """One exact entry lifecycle prepared by a fail-closed ledger adapter.

    A resolved ``net_realized_jpy`` is the audited complete lifecycle: partial
    reductions plus terminal close plus financing.  It is valid only with the
    ``ALL_AUDITED_EXITS`` scope, broker close event UID, close timestamp, and
    outcome evidence digest.  All five fields must be absent for an unresolved
    entry, which stays in the first-N cohort so a later resolved trade cannot
    replace it.  The outcome digest names the exact immutable reconciliation
    bytes covering those all-audited cash components.
    ``intent_artifact_sha256`` names the exact immutable intent artifact bytes;
    validating those bytes is the external adapter's responsibility.
    """

    trade_id: str
    exact_lane_id: str
    policy_generation_id: str | None
    activation_sha256: str | None
    entry_event_uid: str
    intent_generated_at_utc: str
    intent_artifact_sha256: str
    gateway_sent_at_utc: str
    broker_entry_at_utc: str
    broker_close_at_utc: str | None = None
    net_realized_jpy: float | None = None
    outcome_scope: str | None = None
    broker_close_event_uid: str | None = None
    outcome_evidence_sha256: str | None = None


def _snapshot_activation_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return dict(value)
    except Exception as exc:
        raise ValueError("policy-generation activation snapshot is unreadable") from exc


def _activation_keys_are_exact_strings(value: Mapping[object, object]) -> bool:
    return all(_is_exact_str(key) for key in value)


def _is_exact_str(value: object) -> bool:
    return value.__class__ is str


def _is_exact_number(value: object) -> bool:
    return value.__class__ in {int, float}


def _is_exact_int(value: object) -> bool:
    return value.__class__ is int


def _snapshot_observation(
    value: GenerationScopedOutcomeObservation,
) -> GenerationScopedOutcomeObservation:
    if value.__class__ is not GenerationScopedOutcomeObservation:
        raise TypeError("observation must use the exact frozen dataclass")
    body = {field.name: getattr(value, field.name) for field in fields(value)}
    return GenerationScopedOutcomeObservation(**body)


def seal_policy_generation_activation(
    *,
    policy_generation_id: str,
    exact_lane_id: str,
    activated_at_utc: str,
    deployment_artifact_sha256: str,
    source_revision: str,
) -> dict[str, Any]:
    """Build a canonical, content-addressed diagnostic activation record."""

    normalized_at = _normalize_utc(activated_at_utc)
    body: dict[str, Any] = {
        "contract": POLICY_GENERATION_SHADOW_CONTRACT,
        "policy_generation_id": policy_generation_id,
        "exact_lane_id": exact_lane_id,
        "activated_at_utc": normalized_at,
        "deployment_artifact_sha256": deployment_artifact_sha256,
        "source_revision": source_revision,
        "read_only": True,
        "live_permission_allowed": False,
    }
    provisional = {**body, "activation_sha256": "0" * 64}
    issues = _activation_issues(provisional, verify_digest=False)
    if issues:
        raise ValueError("invalid policy-generation activation: " + "; ".join(issues))
    return {
        **body,
        "activation_sha256": policy_generation_activation_sha256(body),
    }


def policy_generation_activation_sha256(
    activation: Mapping[str, Any],
) -> str:
    """Return the canonical digest of the activation body."""

    if not isinstance(activation, Mapping):
        raise TypeError("activation must be a mapping")
    snapshot = _snapshot_activation_mapping(activation)
    if not _activation_keys_are_exact_strings(snapshot):
        raise ValueError("activation keys must be exact strings")
    body = {key: snapshot.get(key) for key in sorted(_ACTIVATION_BODY_KEYS)}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_policy_generation_activation(
    activation: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return validation issues; an empty tuple means the seal is valid."""

    if not isinstance(activation, Mapping):
        return ("activation must be a mapping",)
    try:
        frozen_activation = _snapshot_activation_mapping(activation)
    except ValueError:
        return ("activation snapshot is unreadable",)
    return tuple(_activation_issues(frozen_activation, verify_digest=True))


def evaluate_policy_generation_shadow(
    *,
    activation: Mapping[str, Any] | None,
    observations: Sequence[GenerationScopedOutcomeObservation],
    cohort_size: int = 20,
    source_stream_complete: bool = False,
) -> dict[str, Any]:
    """Evaluate a generation-scoped first-N cohort without changing live gates.

    The caller must attest that ``observations`` is the complete chronological
    exact-entry stream available to its ledger adapter.  This boolean is not
    permission; it only prevents a convenient hand-picked subset from being
    labelled a valid shadow cohort.  Live use still requires a separately
    audited append-only adapter and is outside this module.
    """

    common = {
        "contract": POLICY_GENERATION_SHADOW_CONTRACT,
        "read_only": True,
        "live_permission_allowed": False,
        "gate_relaxation_allowed": False,
        "current_all_time_gate_authoritative": True,
        "shadow_may_replace_all_time_gate": False,
        # This pure evaluator validates only the supplied content addresses.
        # An external adapter still has to prove those bytes came from the
        # deployed runtime and that the observation stream is complete.
        "independent_deployment_proof_verified_by_evaluator": False,
        "runtime_ledger_adapter_verified_by_evaluator": False,
        "intent_artifact_digest_verified_by_evaluator": False,
        "external_intent_artifact_digest_verification_required": True,
        "outcome_evidence_digest_verified_by_evaluator": False,
        "external_outcome_evidence_digest_verification_required": True,
        "source_stream_completeness_caller_attested": (source_stream_complete is True),
        "independent_source_completeness_verified_by_evaluator": False,
        "activation_boundary_proven_for_live": False,
    }
    frozen_activation: dict[str, Any] | None = None
    if isinstance(activation, Mapping):
        try:
            frozen_activation = _snapshot_activation_mapping(activation)
        except ValueError:
            pass
    activation_issues = (
        tuple(_activation_issues(frozen_activation, verify_digest=True))
        if frozen_activation is not None
        else ("activation must be a readable mapping",)
    )
    if activation_issues:
        return {
            **common,
            "status": "INVALID_ACTIVATION",
            "issues": list(activation_issues),
            "shadow_activation_binding_consistent": False,
            "metrics": None,
        }
    assert frozen_activation is not None
    activation = frozen_activation

    if not _is_exact_int(cohort_size) or not 1 <= cohort_size <= 1000:
        return {
            **common,
            "status": "INVALID_SOURCE",
            "issues": ["cohort_size must be an integer in 1..1000"],
            "shadow_activation_binding_consistent": False,
            "metrics": None,
        }
    if source_stream_complete is not True:
        return {
            **common,
            "status": "SOURCE_STREAM_INCOMPLETE",
            "issues": [
                "complete exact-entry stream is required; a hand-picked subset is not a cohort"
            ],
            "shadow_activation_binding_consistent": False,
            "metrics": None,
        }
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        return {
            **common,
            "status": "INVALID_SOURCE",
            "issues": ["observations must be a sequence"],
            "shadow_activation_binding_consistent": False,
            "metrics": None,
        }
    try:
        frozen_observations = tuple(observations)
    except Exception:
        return {
            **common,
            "status": "INVALID_SOURCE",
            "issues": ["observations stream is unreadable"],
            "shadow_activation_binding_consistent": False,
            "metrics": None,
        }

    parsed: list[
        tuple[
            GenerationScopedOutcomeObservation,
            _UtcInstant,
            _UtcInstant,
            _UtcInstant,
            _UtcInstant | None,
        ]
    ] = []
    source_issues: list[str] = []
    trade_ids: set[str] = set()
    event_uids: set[str] = set()
    outcome_evidence_shas: set[str] = set()
    for index, raw_observation in enumerate(frozen_observations):
        if raw_observation.__class__ is not GenerationScopedOutcomeObservation:
            source_issues.append(f"observation[{index}] has the wrong type")
            continue
        try:
            observation = _snapshot_observation(raw_observation)
        except Exception:
            source_issues.append(f"observation[{index}] is unreadable")
            continue
        issues, intent_at, gateway_at, entry_at, close_at = _observation_issues(
            observation
        )
        source_issues.extend(f"observation[{index}]: {issue}" for issue in issues)
        trade_id = observation.trade_id if _is_exact_str(observation.trade_id) else ""
        if trade_id in trade_ids:
            source_issues.append(f"observation[{index}]: duplicate trade_id {trade_id}")
        trade_ids.add(trade_id)
        entry_event_uid = (
            observation.entry_event_uid
            if _is_exact_str(observation.entry_event_uid)
            else ""
        )
        if entry_event_uid in event_uids:
            source_issues.append(
                f"observation[{index}]: duplicate entry_event_uid " f"{entry_event_uid}"
            )
        event_uids.add(entry_event_uid)
        broker_close_event_uid = observation.broker_close_event_uid
        if _is_exact_str(broker_close_event_uid):
            if broker_close_event_uid in event_uids:
                source_issues.append(
                    f"observation[{index}]: duplicate broker_close_event_uid "
                    f"{broker_close_event_uid}"
                )
            event_uids.add(broker_close_event_uid)
        outcome_evidence_sha = observation.outcome_evidence_sha256
        if _is_exact_str(outcome_evidence_sha):
            if outcome_evidence_sha in outcome_evidence_shas:
                source_issues.append(
                    f"observation[{index}]: duplicate outcome_evidence_sha256 "
                    f"{outcome_evidence_sha}"
                )
            outcome_evidence_shas.add(outcome_evidence_sha)
        if (
            not issues
            and intent_at is not None
            and gateway_at is not None
            and entry_at is not None
        ):
            parsed.append((observation, intent_at, gateway_at, entry_at, close_at))
    if source_issues:
        return {
            **common,
            "status": "INVALID_SOURCE",
            "issues": source_issues,
            "shadow_activation_binding_consistent": False,
            "metrics": None,
        }

    activated_at = _parse_utc(str(activation["activated_at_utc"]))
    assert activated_at is not None
    exact_lane_id = str(activation["exact_lane_id"])
    generation_id = str(activation["policy_generation_id"])
    activation_sha = str(activation["activation_sha256"])

    out_of_scope = 0
    pre_activation = 0
    pre_activation_intent_post_activation_gateway_fill = 0
    pre_activation_gateway_post_activation_fill = 0
    unbound_post_activation: list[str] = []
    eligible: list[
        tuple[
            GenerationScopedOutcomeObservation,
            _UtcInstant,
            _UtcInstant,
            _UtcInstant,
            _UtcInstant | None,
        ]
    ] = []
    for item in parsed:
        observation, intent_at, gateway_at, entry_at, _ = item
        if observation.exact_lane_id != exact_lane_id:
            out_of_scope += 1
            continue
        if (
            intent_at <= activated_at
            or gateway_at <= activated_at
            or entry_at <= activated_at
        ):
            pre_activation += 1
            if intent_at <= activated_at < gateway_at and activated_at < entry_at:
                pre_activation_intent_post_activation_gateway_fill += 1
            if gateway_at <= activated_at < entry_at:
                pre_activation_gateway_post_activation_fill += 1
            continue
        if (
            observation.policy_generation_id != generation_id
            or observation.activation_sha256 != activation_sha
        ):
            unbound_post_activation.append(observation.trade_id)
            continue
        eligible.append(item)

    counts = {
        "observations": len(parsed),
        "out_of_scope": out_of_scope,
        "pre_activation_exact_lane": pre_activation,
        "pre_activation_intent_post_activation_gateway_fill": (
            pre_activation_intent_post_activation_gateway_fill
        ),
        "pre_activation_gateway_post_activation_fill": (
            pre_activation_gateway_post_activation_fill
        ),
        "unbound_post_activation_exact_lane": len(unbound_post_activation),
        "bound_post_activation_exact_lane": len(eligible),
    }
    if unbound_post_activation:
        return {
            **common,
            "status": "ACTIVATION_BOUNDARY_UNPROVEN",
            "issues": [
                "one or more post-activation exact-lane entries lack the exact "
                "generation/activation binding"
            ],
            "shadow_activation_binding_consistent": False,
            "policy_generation_id": generation_id,
            "exact_lane_id": exact_lane_id,
            "activation_sha256": activation_sha,
            "counts": counts,
            "unbound_trade_ids_sha256": _trade_ids_sha256(unbound_post_activation),
            "metrics": None,
        }

    eligible.sort(
        key=lambda item: (
            item[3],
            item[2],
            item[0].entry_event_uid,
            item[0].trade_id,
        )
    )
    frozen = eligible[:cohort_size]
    unresolved = [item for item in frozen if item[0].net_realized_jpy is None]
    resolved = [item for item in frozen if item[0].net_realized_jpy is not None]
    pnl = [float(item[0].net_realized_jpy) for item in resolved]
    wins = sum(value > 0.0 for value in pnl)
    losses = sum(value < 0.0 for value in pnl)
    flats = sum(value == 0.0 for value in pnl)
    try:
        gross_profit = math.fsum(value for value in pnl if value > 0.0)
        gross_loss = -math.fsum(value for value in pnl if value < 0.0)
        net = math.fsum(pnl)
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else None
    except (OverflowError, ValueError, ZeroDivisionError):
        return {
            **common,
            "status": "INVALID_SOURCE",
            "issues": ["generation outcome economics overflow"],
            "shadow_activation_binding_consistent": False,
            "policy_generation_id": generation_id,
            "exact_lane_id": exact_lane_id,
            "activation_sha256": activation_sha,
            "counts": counts,
            "metrics": None,
        }
    if not all(math.isfinite(value) for value in (gross_profit, gross_loss, net)) or (
        profit_factor is not None and not math.isfinite(profit_factor)
    ):
        return {
            **common,
            "status": "INVALID_SOURCE",
            "issues": ["generation outcome economics are non-finite"],
            "shadow_activation_binding_consistent": False,
            "policy_generation_id": generation_id,
            "exact_lane_id": exact_lane_id,
            "activation_sha256": activation_sha,
            "counts": counts,
            "metrics": None,
        }
    metrics = {
        "target_entries": cohort_size,
        "frozen_entries": len(frozen),
        "resolved_entries": len(resolved),
        "unresolved_entries": len(unresolved),
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "net_realized_jpy": round(net, 4),
        "expectancy_net_realized_jpy_per_resolved_entry": (
            round(net / len(resolved), 4) if resolved else None
        ),
        "profit_factor": (
            round(profit_factor, 6) if profit_factor is not None else None
        ),
        "first_n_trade_ids_sha256": _trade_ids_sha256(
            [item[0].trade_id for item in frozen]
        ),
    }
    if len(frozen) < cohort_size:
        status = "COLLECTING_GENERATION_EVIDENCE"
    elif unresolved:
        status = "WAITING_FOR_FIRST_N_RESOLUTION"
    else:
        status = "READY_FOR_DIAGNOSTIC_REVIEW"
    return {
        **common,
        "status": status,
        "issues": [],
        "shadow_activation_binding_consistent": True,
        "policy_generation_id": generation_id,
        "exact_lane_id": exact_lane_id,
        "activation_sha256": activation_sha,
        "counts": counts,
        "metrics": metrics,
    }


def _activation_issues(
    activation: Mapping[str, Any], *, verify_digest: bool
) -> list[str]:
    issues: list[str] = []
    if not _activation_keys_are_exact_strings(activation):
        issues.append("activation keys must be exact strings")
    if set(activation) != _ACTIVATION_KEYS:
        issues.append("activation keys do not match the closed schema")
    contract = activation.get("contract")
    if not _is_exact_str(contract) or contract != POLICY_GENERATION_SHADOW_CONTRACT:
        issues.append("activation contract is unsupported")
    generation_id = activation.get("policy_generation_id")
    if not _is_exact_str(generation_id) or not _GENERATION_RE.fullmatch(generation_id):
        issues.append("policy_generation_id is invalid")
    lane_id = activation.get("exact_lane_id")
    if not _is_exact_str(lane_id) or not _valid_exact_lane_id(lane_id):
        issues.append("exact_lane_id must be a canonical five-part lane")
    activated_at = activation.get("activated_at_utc")
    normalized_at = _normalize_utc_or_none(activated_at)
    if normalized_at is None or activated_at != normalized_at:
        issues.append("activated_at_utc must be canonical UTC")
    deployment_sha = activation.get("deployment_artifact_sha256")
    if not _is_exact_str(deployment_sha) or not _SHA256_RE.fullmatch(deployment_sha):
        issues.append("deployment_artifact_sha256 is invalid")
    revision = activation.get("source_revision")
    if not _is_exact_str(revision) or not _REVISION_RE.fullmatch(revision):
        issues.append("source_revision is invalid")
    if activation.get("read_only") is not True:
        issues.append("activation must be read_only")
    if activation.get("live_permission_allowed") is not False:
        issues.append("activation must deny live permission")
    digest = activation.get("activation_sha256")
    if not _is_exact_str(digest) or not _SHA256_RE.fullmatch(digest):
        issues.append("activation_sha256 is invalid")
    elif verify_digest:
        try:
            computed_digest = policy_generation_activation_sha256(activation)
        except (TypeError, ValueError):
            issues.append("activation body is not canonical JSON")
        else:
            if digest != computed_digest:
                issues.append("activation_sha256 does not match the canonical body")
    return issues


def _observation_issues(
    observation: GenerationScopedOutcomeObservation,
) -> tuple[
    list[str],
    _UtcInstant | None,
    _UtcInstant | None,
    _UtcInstant | None,
    _UtcInstant | None,
]:
    issues: list[str] = []
    trade_id = observation.trade_id
    if not _is_exact_str(trade_id) or not _LANE_TOKEN_RE.fullmatch(trade_id):
        issues.append("trade_id is invalid")
    entry_event_uid = observation.entry_event_uid
    if not _is_exact_str(entry_event_uid) or not _EVENT_UID_RE.fullmatch(
        entry_event_uid
    ):
        issues.append("entry_event_uid is invalid")
    if not _valid_exact_lane_id(observation.exact_lane_id):
        issues.append("exact_lane_id is invalid")
    generation_id = observation.policy_generation_id
    if generation_id is not None and (
        not _is_exact_str(generation_id) or not _GENERATION_RE.fullmatch(generation_id)
    ):
        issues.append("policy_generation_id is invalid")
    activation_sha = observation.activation_sha256
    if activation_sha is not None and (
        not _is_exact_str(activation_sha) or not _SHA256_RE.fullmatch(activation_sha)
    ):
        issues.append("activation_sha256 is invalid")
    intent_artifact_sha = observation.intent_artifact_sha256
    if not _is_exact_str(intent_artifact_sha) or not _SHA256_RE.fullmatch(
        intent_artifact_sha
    ):
        issues.append("intent_artifact_sha256 is invalid")
    intent_at = _parse_utc(observation.intent_generated_at_utc)
    gateway_at = _parse_utc(observation.gateway_sent_at_utc)
    entry_at = _parse_utc(observation.broker_entry_at_utc)
    close_at = (
        _parse_utc(observation.broker_close_at_utc)
        if observation.broker_close_at_utc is not None
        else None
    )
    if intent_at is None:
        issues.append("intent_generated_at_utc is invalid")
    if gateway_at is None:
        issues.append("gateway_sent_at_utc is invalid")
    if entry_at is None:
        issues.append("broker_entry_at_utc is invalid")
    if intent_at is not None and gateway_at is not None and gateway_at < intent_at:
        issues.append("gateway send precedes intent generation")
    if gateway_at is not None and entry_at is not None and entry_at < gateway_at:
        issues.append("broker entry precedes gateway send")
    net_realized = observation.net_realized_jpy
    resolved_values = (
        observation.broker_close_at_utc,
        net_realized,
        observation.outcome_scope,
        observation.broker_close_event_uid,
        observation.outcome_evidence_sha256,
    )
    present_count = sum(value is not None for value in resolved_values)
    if present_count not in {0, len(resolved_values)}:
        issues.append("resolved outcome fields must be all present or all absent")
    resolved = present_count == len(resolved_values)
    if resolved and (
        not _is_exact_str(observation.outcome_scope)
        or observation.outcome_scope != "ALL_AUDITED_EXITS"
    ):
        issues.append("outcome_scope must be ALL_AUDITED_EXITS")
    close_event_uid = observation.broker_close_event_uid
    if resolved and (
        not _is_exact_str(close_event_uid)
        or not _EVENT_UID_RE.fullmatch(close_event_uid)
    ):
        issues.append("broker_close_event_uid is invalid")
    outcome_evidence_sha = observation.outcome_evidence_sha256
    if resolved and (
        not _is_exact_str(outcome_evidence_sha)
        or not _SHA256_RE.fullmatch(outcome_evidence_sha)
    ):
        issues.append("outcome_evidence_sha256 is invalid")
    if resolved:
        try:
            net_realized_finite = _is_exact_number(net_realized) and math.isfinite(
                float(net_realized)
            )
        except (OverflowError, TypeError, ValueError):
            net_realized_finite = False
        if not net_realized_finite:
            issues.append("net_realized_jpy is invalid")
    if resolved and close_at is None:
        issues.append("broker_close_at_utc is invalid")
    if close_at is not None and entry_at is not None and close_at < entry_at:
        issues.append("broker close precedes broker entry")
    return issues, intent_at, gateway_at, entry_at, close_at


def _valid_exact_lane_id(value: object) -> bool:
    if not _is_exact_str(value) or value != value.strip():
        return False
    parts = value.split(":")
    if len(parts) != 5:
        return False
    desk, pair, side, method, vehicle = parts
    return bool(
        _LANE_TOKEN_RE.fullmatch(desk)
        and _PAIR_RE.fullmatch(pair)
        and side in {"LONG", "SHORT"}
        and _LANE_TOKEN_RE.fullmatch(method)
        and method == method.upper()
        and vehicle in {"LIMIT", "MARKET", "STOP"}
    )


def _normalize_utc(value: object) -> str:
    normalized = _normalize_utc_or_none(value)
    if normalized is None:
        raise ValueError("timestamp must be a timezone-aware RFC3339 instant")
    return normalized


def _normalize_utc_or_none(value: object) -> str | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    return parsed.canonical()


def _parse_utc(value: object) -> _UtcInstant | None:
    if not _is_exact_str(value) or not value or value != value.strip():
        return None
    match = _RFC3339_UTC_RE.fullmatch(value)
    if match is None:
        return None
    try:
        parsed_second = datetime.strptime(
            f"{match.group('date')}T{match.group('clock')}",
            "%Y-%m-%dT%H:%M:%S",
        )
    except ValueError:
        return None
    fraction = match.group("fraction") or ""
    nanosecond = int(fraction.ljust(9, "0")) if fraction else 0
    return _UtcInstant(
        epoch_second=int(parsed_second.replace(tzinfo=timezone.utc).timestamp()),
        nanosecond=nanosecond,
    )


def _trade_ids_sha256(trade_ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(trade_ids),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
