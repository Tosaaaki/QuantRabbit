from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MARKET_READ_OVERLAY_SCHEMA_VERSION = 1
CODEX_MARKET_READ_AUTHOR = "CODEX_MARKET_READ"
DETERMINISTIC_BASELINE_AUTHOR = "DETERMINISTIC_BASELINE"
DEFAULT_OVERLAY_MAX_AGE_SECONDS = 15 * 60

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
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def market_read_sha256(market_read: Any) -> str:
    return canonical_json_sha256(market_read if isinstance(market_read, dict) else {})


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
    """Fields an AI veto cannot change.

    The merge tool itself may downgrade TRADE to WAIT/REQUEST_EVIDENCE and
    clear the selected lane so the non-trade receipt cannot reach a gateway.
    Everything else, including units embedded in cited intents, risk notes,
    cancel/close ids, and evidence refs, remains byte-equivalent to baseline.
    """

    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"action", "selected_lane_id", "selected_lane_ids"}
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
    )
    stored_packet_sha = str(packet.get("evidence_packet_sha256") or "")
    rebuilt_packet_sha = str(rebuilt["evidence_packet_sha256"])
    _require_sha_match(
        actual=rebuilt_packet_sha,
        claimed=stored_packet_sha,
        code="MARKET_READ_EVIDENCE_PACKET_STALE",
        label="stored evidence packet",
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


def revalidate_codex_market_read_artifacts(
    *,
    final_payload: Mapping[str, Any],
    baseline_path: Path,
    packet_path: Path,
    overlay_path: Path,
    evidence_sources: Mapping[str, Path],
    max_overlay_age_seconds: int = DEFAULT_OVERLAY_MAX_AGE_SECONDS,
) -> list[tuple[str, str]]:
    """Rebuild a TRADE receipt from its actual handoff artifacts.

    Provenance fields inside the final JSON are claims, not proof.  The
    verifier therefore re-runs the same content-addressed merge against the
    current baseline, stored packet, overlay, and every named evidence source,
    then requires the rebuilt final object to be identical.  Non-TRADE
    receipts retain their existing verifier behavior; a veto cannot grant
    entry permission and does not need this extra execution-boundary check.
    """

    if str(final_payload.get("action") or "").strip().upper() != "TRADE":
        return []
    provenance = final_payload.get("decision_provenance")
    if not isinstance(provenance, Mapping):
        return [
            (
                "AI_MARKET_READ_ARTIFACT_PROVENANCE_MISSING",
                "TRADE cannot bind market-read artifacts without decision_provenance",
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
                f"could not rebuild the final TRADE receipt from market-read artifacts: {exc}",
            )
        ]

    rebuilt_sha = canonical_json_sha256(rebuilt)
    final_sha = canonical_json_sha256(dict(final_payload))
    if rebuilt_sha != final_sha:
        return [
            (
                "AI_MARKET_READ_ARTIFACT_FINAL_MISMATCH",
                "final TRADE receipt does not exactly match the current baseline/evidence/overlay merge "
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
    for key in (
        "baseline_sha256",
        "evidence_packet_sha256",
        "overlay_sha256",
        "market_read_sha256",
        "execution_envelope_sha256",
        "baseline_execution_envelope_sha256",
        "final_execution_envelope_sha256",
    ):
        if not SHA256_RE.fullmatch(str(claimed.get(key) or "")):
            issues.append(("AI_MARKET_READ_PROVENANCE_INVALID", f"{key} must be a sha256 digest"))
    if claimed.get("model") != "gpt-5.5" or str(claimed.get("reasoning_effort") or "").lower() != "high":
        issues.append(
            (
                "AI_MARKET_READ_PROVENANCE_INVALID",
                "CODEX_MARKET_READ provenance must name gpt-5.5 with high reasoning effort",
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
    broker = _load_json_object(Path(broker_path), label="broker snapshot")
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
) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for name in sorted(evidence_sources):
        path = Path(evidence_sources[name])
        sources[name] = _source_descriptor(path)
    predictions_path = evidence_sources.get("market_read_predictions")
    recent_predictions = (
        _recent_resolved_predictions(Path(predictions_path))
        if predictions_path is not None
        else []
    )
    material_sources: dict[str, dict[str, Any]] = {}
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
        material_sources[name] = {
            "path": item["path"],
            "exists": item["exists"],
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        }
    material = {
        "schema_version": MARKET_READ_OVERLAY_SCHEMA_VERSION,
        "baseline_sha256": baseline_sha256,
        "sources": material_sources,
    }
    packet_sha = canonical_json_sha256(material)
    broker_path = evidence_sources.get("broker_snapshot")
    broker = _load_optional_json_object(Path(broker_path)) if broker_path is not None else {}
    overrides_path = evidence_sources.get("trader_overrides")
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
        "source_paths": {name: str(path) for name, path in sorted(evidence_sources.items())},
        "source_metadata": sources,
        "quote_basis_by_pair": quote_basis_by_pair_from_broker_payload(broker),
        "prior_market_read_review": (
            overrides.get("market_read_review")
            if isinstance(overrides.get("market_read_review"), dict)
            else {}
        ),
        "recent_resolved_predictions": recent_predictions,
        "contract": {
            "overlay_author": CODEX_MARKET_READ_AUTHOR,
            "allowed_overlay_fields": sorted(OVERLAY_ALLOWED_FIELDS),
            "execution_fields_are_immutable": True,
            "overlay_grants_live_permission": False,
            "directional_and_range_geometry_must_be_numeric": True,
        },
    }


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
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
