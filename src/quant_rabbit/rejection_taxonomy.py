"""Candidate rejection taxonomy (weakness ledger W18).

Permanent shadow cells collect outcomes but nothing records *why* a
candidate died, so failures never feed the next family design.  This module
makes a death code mandatory and machine-aggregable: a fixed vocabulary,
sealed per-rejection records bound to the evidence that killed the
candidate, and a quarterly aggregation that names the dominant death codes
per family.  Recording a death changes no authority and deletes nothing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

CONTRACT = "QR_CANDIDATE_REJECTION_RECORD_V1"
AGGREGATE_CONTRACT = "QR_REJECTION_TAXONOMY_AGGREGATE_V1"
DEATH_CODES = frozenset(
    {
        "COST_SPREAD_DOMINATES_EDGE",
        "DIRECTION_WRONG",
        "EXIT_TIMING_LEFT_PROFIT",
        "ENTRY_TOO_LATE",
        "REGIME_MISMATCH",
        "CLOSE_CROSSING_LEAK",
        "INSUFFICIENT_SAMPLE",
        "ROBUSTNESS_FLOOR_FAILED",
        "MULTIPLICITY_NOT_SIGNIFICANT",
        "EXECUTION_INFEASIBLE",
        "DATA_INTEGRITY_FAILED",
    }
)


class RejectionTaxonomyError(ValueError):
    """Raised when a rejection record is malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_rejection_record(
    *,
    candidate_id: str,
    family_id: str,
    death_code: str,
    evidence_sha256: str,
    rejected_at_utc: datetime,
    note: str = "",
) -> dict[str, Any]:
    """Seal one candidate's cause of death bound to its killing evidence."""

    candidate = str(candidate_id).strip()
    family = str(family_id).strip().upper()
    if not candidate or not family:
        raise RejectionTaxonomyError("candidate and family identity are required")
    code = str(death_code).strip().upper()
    if code not in DEATH_CODES:
        raise RejectionTaxonomyError(f"death code is not in the fixed taxonomy: {code}")
    evidence = str(evidence_sha256 or "")
    if len(evidence) != 64 or any(c not in "0123456789abcdef" for c in evidence):
        raise RejectionTaxonomyError("evidence_sha256 must be a lowercase sha256")
    if rejected_at_utc.tzinfo is None:
        raise RejectionTaxonomyError("rejection clock must be timezone-aware")
    body = {
        "contract": CONTRACT,
        "schema_version": 1,
        "candidate_id": candidate,
        "family_id": family,
        "death_code": code,
        "evidence_sha256": evidence,
        "rejected_at_utc": rejected_at_utc.astimezone(timezone.utc).isoformat(),
        "note": str(note),
        "candidate_shadow_retained": True,
        "order_authority": "NONE",
    }
    return {**body, "record_sha256": _canonical_sha(body)}


def aggregate_rejections(
    records: Sequence[Mapping[str, Any]],
    *,
    window_label: str,
) -> dict[str, Any]:
    """Count death codes per family so the next family design targets them."""

    if not records:
        raise RejectionTaxonomyError("at least one rejection record is required")
    counts: dict[str, dict[str, int]] = {}
    for index, record in enumerate(records):
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        if record.get("record_sha256") != _canonical_sha(body):
            raise RejectionTaxonomyError(f"record {index} digest is invalid")
        if record.get("contract") != CONTRACT:
            raise RejectionTaxonomyError(f"record {index} contract is invalid")
        family = str(record["family_id"])
        code = str(record["death_code"])
        counts.setdefault(family, {})
        counts[family][code] = counts[family].get(code, 0) + 1

    family_rows = []
    for family in sorted(counts):
        codes = counts[family]
        ranked = sorted(codes.items(), key=lambda item: (-item[1], item[0]))
        family_rows.append(
            {
                "family_id": family,
                "total_rejections": sum(codes.values()),
                "death_code_counts": dict(ranked),
                "dominant_death_code": ranked[0][0],
            }
        )
    body = {
        "contract": AGGREGATE_CONTRACT,
        "schema_version": 1,
        "window_label": str(window_label),
        "record_count": len(records),
        "family_rows": family_rows,
        "measurement_only": True,
        "order_authority": "NONE",
    }
    return {**body, "aggregate_sha256": _canonical_sha(body)}
