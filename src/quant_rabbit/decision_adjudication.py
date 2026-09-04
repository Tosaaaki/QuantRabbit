"""Deterministic arbitration for sealed AI entry and exit decisions.

This module deliberately owns no broker behavior.  It validates the immutable
decision identities and shared resource claims, selects at most one mutation,
and emits a content-addressed receipt for a later deterministic gateway.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


ADJUDICATION_CONTRACT = "quant_rabbit.decision_adjudication.v1"
ADJUDICATION_SCHEMA_VERSION = 1

ENTRY_ID_PREFIX = "qre_"
EXIT_ID_PREFIX = "qrx_"
ADJUDICATION_ID_PREFIX = "qra_"

ENTRY_ACTIONS = frozenset({"ENTER", "WAIT", "REQUEST_EVIDENCE"})
EXIT_ACTIONS = frozenset(
    {"HOLD", "CLOSE_ALL", "REDUCE", "TIGHTEN_SL", "REPLACE_TP", "REQUEST_EVIDENCE"}
)
EXIT_MUTATIONS = frozenset({"CLOSE_ALL", "REDUCE", "TIGHTEN_SL", "REPLACE_TP"})
EMERGENCY_EXIT_ACTIONS = frozenset({"CLOSE_ALL", "REDUCE"})
NON_MUTATING_ACTIONS = frozenset({"WAIT", "HOLD", "REQUEST_EVIDENCE"})

_CONTENT_ID_RE = re.compile(r"^(?:qre|qrx)_[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_CLAIM_KINDS = frozenset({"account", "pair", "campaign", "position", "order"})


class DecisionAdjudicationError(ValueError):
    """A sealed input was invalid, stale, or outside the requested cycle."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Proposal:
    decision_id: str
    kind: str
    action: str
    claims: tuple[str, ...]
    pair: str | None
    side: str | None
    position_side: str | None
    owner_kind: str | None
    owner_binding_complete: bool
    emergency_eligible: bool

    @property
    def mutates(self) -> bool:
        return self.action == "ENTER" or self.action in EXIT_MUTATIONS

    @property
    def priority(self) -> int:
        if self.kind == "EXIT" and self.emergency_eligible:
            return 0
        if self.kind == "EXIT" and self.mutates:
            return 1
        if self.kind == "ENTRY" and self.mutates:
            return 2
        return 3


def canonical_content_id(value: Mapping[str, Any], *, prefix: str) -> str:
    """Return the canonical identity used by Entry, Exit, and test producers."""

    if prefix not in {ENTRY_ID_PREFIX, EXIT_ID_PREFIX, ADJUDICATION_ID_PREFIX}:
        raise DecisionAdjudicationError("ID_PREFIX_INVALID", f"unsupported content-id prefix: {prefix}")
    material = dict(value)
    if prefix in {ENTRY_ID_PREFIX, EXIT_ID_PREFIX}:
        material.pop("decision_id", None)
    else:
        material.pop("adjudication_id", None)
    return prefix + hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def adjudicate_decisions(
    *,
    cycle_id: str,
    broker_epoch: str,
    entry_proposals: Sequence[Mapping[str, Any]] = (),
    exit_proposals: Sequence[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate sealed proposals and select at most one broker mutation.

    A validation failure raises before a receipt is emitted.  Ineligible
    ownership and non-mutating actions are recorded as deterministic rejection
    reasons because they are valid observations, not malformed inputs.
    """

    expected_cycle = _opaque_binding(cycle_id, "cycle_id")
    expected_epoch = _opaque_binding(broker_epoch, "broker_epoch")
    current = _utc_now(now)

    proposals = [
        *(
            _validate_proposal(
                raw,
                kind="ENTRY",
                expected_cycle=expected_cycle,
                expected_epoch=expected_epoch,
                now=current,
            )
            for raw in entry_proposals
        ),
        *(
            _validate_proposal(
                raw,
                kind="EXIT",
                expected_cycle=expected_cycle,
                expected_epoch=expected_epoch,
                now=current,
            )
            for raw in exit_proposals
        ),
    ]
    identifiers = [item.decision_id for item in proposals]
    if len(identifiers) != len(set(identifiers)):
        raise DecisionAdjudicationError(
            "DUPLICATE_PROPOSAL_ID",
            "the same sealed proposal identity was supplied more than once",
        )

    ordered = sorted(proposals, key=lambda item: (item.priority, item.decision_id))
    selected: _Proposal | None = None
    rejected: list[dict[str, Any]] = []

    for proposal in ordered:
        ineligible_reason = _ineligible_reason(proposal)
        if ineligible_reason is not None:
            rejected.append(_rejection(proposal, ineligible_reason))
            continue
        if selected is None:
            selected = proposal
            continue

        reverse_reason = _reverse_after_exit_reason(selected, proposal)
        if reverse_reason is not None:
            rejected.append(_rejection(proposal, reverse_reason))
            continue
        conflicts = sorted(set(selected.claims) & set(proposal.claims))
        if conflicts:
            rejected.append(
                _rejection(proposal, "RESOURCE_CONFLICT", conflicting_claims=conflicts)
            )
            continue
        rejected.append(_rejection(proposal, "MAX_ONE_MUTATION_PER_CYCLE"))

    rejected.sort(key=lambda item: (str(item["proposal_id"]), str(item["reason"])))
    selected_claims = list(selected.claims) if selected is not None else []
    receipt: dict[str, Any] = {
        "contract": ADJUDICATION_CONTRACT,
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "cycle_id": expected_cycle,
        "broker_epoch": expected_epoch,
        "input_proposal_ids": sorted(identifiers),
        "selected_proposal_id": selected.decision_id if selected is not None else None,
        "selected_kind": selected.kind if selected is not None else None,
        "selected_action": selected.action if selected is not None else None,
        "rejected_proposals": rejected,
        "resource_claims": selected_claims,
        "mutation_count": 1 if selected is not None else 0,
        "require_fresh_broker_readback_after_mutation": selected is not None,
    }
    receipt["adjudication_id"] = canonical_content_id(
        receipt,
        prefix=ADJUDICATION_ID_PREFIX,
    )
    return receipt


def verify_adjudication_receipt(receipt: Mapping[str, Any]) -> None:
    """Fail closed unless a receipt is a valid content-addressed qra mapping."""

    recorded = str(receipt.get("adjudication_id") or "")
    expected = canonical_content_id(receipt, prefix=ADJUDICATION_ID_PREFIX)
    if recorded != expected:
        raise DecisionAdjudicationError(
            "ADJUDICATION_ID_MISMATCH",
            "adjudication_id does not match the canonical receipt content",
        )


def _validate_proposal(
    raw: Mapping[str, Any],
    *,
    kind: str,
    expected_cycle: str,
    expected_epoch: str,
    now: datetime,
) -> _Proposal:
    if not isinstance(raw, Mapping):
        raise DecisionAdjudicationError("PROPOSAL_INVALID", f"{kind} proposal must be a mapping")
    proposal = _validate_source_decision(
        raw,
        kind=kind,
        expected_cycle=expected_cycle,
        expected_epoch=expected_epoch,
        now=now,
    )
    if proposal.get("schema_version") != 1:
        raise DecisionAdjudicationError(
            "PROPOSAL_SCHEMA_INVALID",
            f"{kind} proposal has an unsupported schema_version",
        )
    decision_id = str(proposal.get("decision_id") or "").strip()
    prefix = ENTRY_ID_PREFIX if kind == "ENTRY" else EXIT_ID_PREFIX
    if not _CONTENT_ID_RE.fullmatch(decision_id) or not decision_id.startswith(prefix):
        raise DecisionAdjudicationError(
            "PROPOSAL_ID_INVALID",
            f"{kind} proposal has an invalid decision_id",
        )
    expected_id = canonical_content_id(proposal, prefix=prefix)
    if decision_id != expected_id:
        raise DecisionAdjudicationError(
            "PROPOSAL_ID_MISMATCH",
            f"{kind} decision_id does not match its canonical content",
        )
    if _opaque_binding(proposal.get("cycle_id"), "proposal cycle_id") != expected_cycle:
        raise DecisionAdjudicationError(
            "CYCLE_ID_MISMATCH",
            f"{kind} proposal belongs to a different cycle",
        )
    if _opaque_binding(proposal.get("broker_epoch"), "proposal broker_epoch") != expected_epoch:
        raise DecisionAdjudicationError(
            "BROKER_EPOCH_MISMATCH",
            f"{kind} proposal belongs to a different broker epoch",
        )
    created_at = _required_utc(proposal.get("created_at_utc"), "created_at_utc")
    expires_at = _required_utc(proposal.get("expires_at_utc"), "expires_at_utc")
    if created_at > expires_at or created_at > now:
        raise DecisionAdjudicationError(
            "TIMESTAMP_INVALID",
            f"{kind} proposal has an invalid decision time window",
        )
    if now > expires_at or (kind == "EXIT" and now == expires_at):
        raise DecisionAdjudicationError("PROPOSAL_STALE", f"{kind} proposal has expired")

    action = str(proposal.get("action") or "").strip().upper()
    allowed = ENTRY_ACTIONS if kind == "ENTRY" else EXIT_ACTIONS
    if action not in allowed:
        raise DecisionAdjudicationError(
            "ACTION_INVALID",
            f"unsupported {kind} action: {action or '<empty>'}",
        )

    nested: dict[str, Any] = {}
    if kind == "ENTRY":
        payloads = proposal.get("proposals")
        if not isinstance(payloads, list) or len(payloads) > 1:
            raise DecisionAdjudicationError(
                "PROPOSAL_INVALID",
                "Entry decision must contain a zero-or-one proposals list",
            )
        if payloads:
            payload = payloads[0]
            if not isinstance(payload, Mapping):
                raise DecisionAdjudicationError(
                    "PROPOSAL_INVALID",
                    "Entry proposal must be a mapping",
                )
            nested = dict(payload)
        if action == "ENTER" and not nested:
            raise DecisionAdjudicationError("PROPOSAL_INVALID", "ENTER requires one nested proposal")
        if action in NON_MUTATING_ACTIONS and nested:
            raise DecisionAdjudicationError(
                "PROPOSAL_INVALID",
                f"{action} must not carry a mutating nested proposal",
            )

    raw_claims = nested.get("resource_claims") if kind == "ENTRY" else proposal.get("resource_claims")
    claims = _validated_claims(raw_claims)
    pair = _pair(nested.get("pair") if kind == "ENTRY" else proposal.get("instrument") or proposal.get("pair"))
    side = _side(nested.get("side")) if kind == "ENTRY" and action == "ENTER" else None
    position_side = _side(proposal.get("position_side"), required=False) if kind == "EXIT" else None

    if action == "ENTER":
        if pair is None or side is None:
            raise DecisionAdjudicationError("PROPOSAL_INVALID", "ENTER requires a valid pair and side")
        campaign_id = _optional_component(nested.get("campaign_id"), "campaign_id")
        _validate_semantic_claims(
            claims,
            pair=pair,
            campaign_id=campaign_id,
            cycle_id=expected_cycle,
        )
        claims = _merge_claims(
            claims,
            (f"pair:{pair}", f"entry:{expected_cycle}:{pair}"),
        )
        if campaign_id is not None:
            claims = _merge_claims(claims, (f"campaign:{campaign_id}",))

    owner_kind: str | None = None
    owner_binding_complete = True
    emergency_eligible = False
    if kind == "EXIT":
        owner_binding = proposal.get("owner_binding")
        owner = dict(owner_binding) if isinstance(owner_binding, Mapping) else {}
        owner_kind = str(owner.get("owner_kind") or proposal.get("owner_kind") or "").strip().upper()
        owner_binding_complete = owner_kind == "AI_SYSTEM" and all(
            _claim_value(str(owner.get(field) or "").strip())
            for field in ("owner_id", "client_extension_id", "campaign_id")
        )
        emergency_raw = proposal.get("emergency_eligible", False)
        if not isinstance(emergency_raw, bool):
            raise DecisionAdjudicationError(
                "PROPOSAL_INVALID",
                "emergency_eligible must be a boolean",
            )
        emergency_eligible = emergency_raw
        if emergency_eligible and action not in EMERGENCY_EXIT_ACTIONS:
            raise DecisionAdjudicationError(
                "EMERGENCY_ACTION_INVALID",
                "emergency eligibility is allowed only for CLOSE_ALL or REDUCE",
            )
        if action in EXIT_MUTATIONS:
            trade_id = _required_component(proposal.get("trade_id"), "trade_id")
            campaign_id = _optional_component(owner.get("campaign_id"), "campaign_id")
            _validate_semantic_claims(
                claims,
                pair=pair,
                campaign_id=campaign_id,
                trade_id=trade_id,
                cycle_id=expected_cycle,
            )
            claims = _merge_claims(claims, (f"position:{trade_id}",))
            if campaign_id is not None:
                claims = _merge_claims(claims, (f"campaign:{campaign_id}",))
            if pair is not None:
                claims = _merge_claims(claims, (f"pair:{pair}",))
                if action in {"CLOSE_ALL", "REDUCE"}:
                    claims = _merge_claims(
                        claims,
                        (f"reverse-entry:{expected_cycle}:{pair}",),
                    )

    return _Proposal(
        decision_id=decision_id,
        kind=kind,
        action=action,
        claims=claims,
        pair=pair,
        side=side,
        position_side=position_side,
        owner_kind=owner_kind,
        owner_binding_complete=owner_binding_complete,
        emergency_eligible=emergency_eligible,
    )


def _validate_source_decision(
    raw: Mapping[str, Any],
    *,
    kind: str,
    expected_cycle: str,
    expected_epoch: str,
    now: datetime,
) -> dict[str, Any]:
    if kind == "ENTRY":
        from quant_rabbit.entry_decision import EntryDecisionError, validate_entry_decision

        try:
            return validate_entry_decision(
                raw,
                expected_cycle_id=expected_cycle,
                expected_broker_epoch=expected_epoch,
                now_utc=now,
            )
        except EntryDecisionError as exc:
            raise DecisionAdjudicationError(
                _source_error_code(exc.code),
                f"invalid sealed Entry decision: {exc}",
            ) from exc

    from quant_rabbit.exit_decision import ExitDecision, ExitDecisionError, NoTouchError

    try:
        return ExitDecision.from_mapping(raw).to_dict()
    except NoTouchError:
        # NO_TOUCH is an eligibility outcome, not permission to ignore sealing.
        # The generic path below still checks the qrx identity, exact cycle and
        # epoch, TTL, action, and resource-claim bindings before rejecting it.
        return dict(raw)
    except ExitDecisionError as exc:
        raise DecisionAdjudicationError(
            _source_error_code(exc.code),
            f"invalid sealed Exit decision: {exc}",
        ) from exc


def _source_error_code(code: str) -> str:
    normalized = str(code or "").strip().upper()
    if normalized in {"CYCLE_MISMATCH", "CYCLE_ID_MISMATCH"}:
        return "CYCLE_ID_MISMATCH"
    if normalized == "BROKER_EPOCH_MISMATCH":
        return normalized
    if normalized in {"DECISION_STALE", "DECISION_EXPIRED", "PROPOSAL_STALE"}:
        return "PROPOSAL_STALE"
    if normalized in {
        "DECISION_ID_MISMATCH",
        "DECISION_TAMPERED",
        "INVALID_DECISION_ID",
        "PROPOSAL_ID_MISMATCH",
    }:
        return "PROPOSAL_ID_MISMATCH"
    return f"SOURCE_{normalized or 'DECISION_INVALID'}"


def _ineligible_reason(proposal: _Proposal) -> str | None:
    if not proposal.mutates:
        return "NON_MUTATING_ACTION"
    if proposal.kind == "EXIT" and not proposal.owner_binding_complete:
        return "NO_TOUCH_OWNER"
    return None


def _reverse_after_exit_reason(selected: _Proposal, candidate: _Proposal) -> str | None:
    if selected.kind != "EXIT" or selected.action not in {"CLOSE_ALL", "REDUCE"}:
        return None
    if candidate.kind != "ENTRY" or candidate.action != "ENTER":
        return None
    if selected.pair is None or candidate.pair != selected.pair:
        return None
    if selected.position_side is None or candidate.side != selected.position_side:
        return "REVERSE_ENTRY_REQUIRES_FRESH_BROKER_READBACK"
    return None


def _rejection(
    proposal: _Proposal,
    reason: str,
    *,
    conflicting_claims: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "proposal_id": proposal.decision_id,
        "kind": proposal.kind,
        "action": proposal.action,
        "reason": reason,
        "conflicting_claims": list(conflicting_claims),
    }


def _validated_claims(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise DecisionAdjudicationError("CLAIMS_INVALID", "resource_claims must be a list")
    claims: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip() or len(item) > 768:
            raise DecisionAdjudicationError("CLAIMS_INVALID", "resource claim must be a bounded string")
        _validate_claim(item)
        if item in claims:
            raise DecisionAdjudicationError("CLAIMS_INVALID", f"duplicate resource claim: {item}")
        claims.append(item)
    return tuple(sorted(claims))


def _validate_claim(claim: str) -> None:
    kind, separator, value = claim.partition(":")
    if kind in {"entry", "reverse-entry"}:
        cycle_and_pair = _cycle_pair_claim(claim, kind=kind)
        if cycle_and_pair is None:
            raise DecisionAdjudicationError("CLAIMS_INVALID", f"invalid resource claim: {claim}")
        return
    if kind == "account":
        if separator and not _claim_value(value):
            raise DecisionAdjudicationError("CLAIMS_INVALID", f"invalid resource claim: {claim}")
        return
    if kind not in _CLAIM_KINDS or not separator or not _claim_value(value):
        raise DecisionAdjudicationError("CLAIMS_INVALID", f"invalid resource claim: {claim}")
    if kind == "pair" and not _PAIR_RE.fullmatch(value):
        raise DecisionAdjudicationError("CLAIMS_INVALID", f"invalid pair claim: {claim}")


def _validate_semantic_claims(
    claims: Sequence[str],
    *,
    pair: str | None,
    campaign_id: str | None,
    trade_id: str | None = None,
    cycle_id: str | None = None,
) -> None:
    for claim in claims:
        if claim.startswith("pair:") and (pair is None or claim != f"pair:{pair}"):
            raise DecisionAdjudicationError(
                "CLAIM_BINDING_MISMATCH",
                "pair resource claim does not match the sealed proposal",
            )
        if (
            claim.startswith("campaign:")
            and (campaign_id is None or claim != f"campaign:{campaign_id}")
        ):
            raise DecisionAdjudicationError(
                "CLAIM_BINDING_MISMATCH",
                "campaign resource claim does not match the sealed proposal",
            )
        if (
            claim.startswith("position:")
            and (trade_id is None or claim != f"position:{trade_id}")
        ):
            raise DecisionAdjudicationError(
                "CLAIM_BINDING_MISMATCH",
                "position resource claim does not match the sealed proposal",
            )
        if claim.startswith("reverse-entry:"):
            if pair is None or cycle_id is None or claim != f"reverse-entry:{cycle_id}:{pair}":
                raise DecisionAdjudicationError(
                    "CLAIM_BINDING_MISMATCH",
                    "reverse-entry resource claim does not match cycle and instrument",
                )
        if claim.startswith("entry:"):
            if pair is None or cycle_id is None or claim != f"entry:{cycle_id}:{pair}":
                raise DecisionAdjudicationError(
                    "CLAIM_BINDING_MISMATCH",
                    "entry resource claim does not match cycle and instrument",
                )


def _cycle_pair_claim(claim: str, *, kind: str) -> tuple[str, str] | None:
    prefix = f"{kind}:"
    if not claim.startswith(prefix):
        return None
    body = claim[len(prefix) :]
    if ":" not in body:
        return None
    cycle_id, pair = body.rsplit(":", 1)
    if (
        not cycle_id
        or cycle_id != cycle_id.strip()
        or len(cycle_id) > 256
        or not _PAIR_RE.fullmatch(pair)
    ):
        return None
    return cycle_id, pair


def _claim_value(value: str) -> bool:
    return bool(
        value
        and value == value.strip()
        and len(value) <= 512
        and "\n" not in value
        and "\r" not in value
    )


def _merge_claims(existing: Sequence[str], derived: Sequence[str]) -> tuple[str, ...]:
    for claim in derived:
        _validate_claim(claim)
    return tuple(sorted(set(existing) | set(derived)))


def _pair(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if not _PAIR_RE.fullmatch(text):
        raise DecisionAdjudicationError("PROPOSAL_INVALID", f"invalid instrument pair: {text}")
    return text


def _side(value: Any, *, required: bool = True) -> str | None:
    text = str(value or "").strip().upper()
    if not text and not required:
        return None
    if text not in {"LONG", "SHORT"}:
        raise DecisionAdjudicationError("PROPOSAL_INVALID", f"invalid side: {text or '<empty>'}")
    return text


def _opaque_binding(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 256:
        raise DecisionAdjudicationError("BINDING_INVALID", f"{label} must be a bounded exact string")
    return value


def _required_component(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not _claim_value(text):
        raise DecisionAdjudicationError("PROPOSAL_INVALID", f"invalid {label}")
    return text


def _optional_component(value: Any, label: str) -> str | None:
    text = str(value or "").strip()
    return _required_component(text, label) if text else None


def _required_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DecisionAdjudicationError("TIMESTAMP_INVALID", f"{label} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DecisionAdjudicationError("TIMESTAMP_INVALID", f"invalid {label}") from exc
    if parsed.tzinfo is None:
        raise DecisionAdjudicationError("TIMESTAMP_INVALID", f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise DecisionAdjudicationError("TIMESTAMP_INVALID", "now must include timezone")
    return current.astimezone(timezone.utc)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecisionAdjudicationError(
            "CANONICAL_CONTENT_INVALID",
            f"content is not canonical JSON: {exc}",
        ) from exc
