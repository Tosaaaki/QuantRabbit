"""Read-only progress for forward guardian tuning evidence acquisition.

The tuning queue can ask the hourly AI to collect immutable pre-entry signal
state before it is allowed to precommit a ``TEST_REQUIRED`` experiment.  This
module reports whether that acquisition has actually happened.  It deliberately
does not calculate economics, suggest a parameter value, or grant execution
permission.

Progress is built from the same canonical gateway-attributed entry and outcome
readers used by capture economics.  For every supported review, the first N
matching entries after the immutable review boundary are frozen before outcomes
are considered.  A later resolved trade therefore cannot replace an earlier
unresolved one.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from quant_rabbit.capture_economics import (
    AttributedEntry,
    RealizedOutcome,
    read_attributed_net_outcomes,
    read_attributed_system_entries,
)


MAX_ENTRY_THESIS_BYTES = 16 * 1024 * 1024
_PENDING_STATUSES = frozenset({"PENDING_HOURLY_AI_REVIEW", "PENDING", "OPEN"})
_FAMILY_BY_METHOD = {
    "TREND_CONTINUATION": "trend",
    "RANGE_ROTATION": "mean_reversion",
    "BREAKOUT_FAILURE": "breakout",
}
_ENTRY_VEHICLES = frozenset({"MARKET", "LIMIT", "STOP"})
_TECHNICAL_STATE_KEYS = frozenset(
    {
        "dominant_regime_state",
        "regime_state",
        "chart_story_structural",
        "chart_direction_bias",
        "m1_regime_quantile",
        "m5_regime",
        "m5_regime_quantile",
        "m15_regime",
        "h1_regime",
        "range_phase",
        "range_breakout_direction",
        "tf_regime_map",
    }
)
_TECHNICAL_FAMILY_KEYS = {
    "trend": frozenset({"m5_trend_score", "trend_timeframes"}),
    "mean_reversion": frozenset({"m5_mean_rev_score", "range_timeframes"}),
    "breakout": frozenset({"m5_breakout_score", "range_breakout_direction"}),
}
_RFC3339_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})"
)


@dataclass(frozen=True)
class _ReviewContract:
    work_order_id: str
    observation_id: str
    boundary_text: str
    boundary_key: tuple[datetime, int]
    pair: str
    family: str
    event_type: str
    required_count: int
    selector_mode: str
    exact_lane_id: str | None


class _DuplicateJsonKey(ValueError):
    pass


def _reject_non_finite_json_number(token: str) -> Any:
    raise ValueError(f"non-finite JSON number: {token}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _instant_key(value: object) -> tuple[datetime, int]:
    """Return a UTC second plus nanoseconds, preserving OANDA precision."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp is required")
    match = _RFC3339_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError("timestamp must be RFC3339 with timezone")
    prefix, fraction, offset = match.groups()
    try:
        parsed = datetime.fromisoformat(
            f"{prefix}{'+00:00' if offset == 'Z' else offset}"
        )
    except ValueError:
        raise ValueError("timestamp is invalid") from None
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return (
        parsed.astimezone(timezone.utc).replace(microsecond=0),
        int((fraction or "0").ljust(9, "0")),
    )


def _normalize_lane(value: object) -> str | None:
    parts = [part.strip() for part in str(value or "").split(":")]
    if (
        len(parts) != 5
        or any(not part for part in parts)
        or parts[2].upper() not in {"LONG", "SHORT"}
        or parts[4].upper() not in _ENTRY_VEHICLES
    ):
        return None
    return ":".join(
        (
            parts[0],
            parts[1].upper(),
            parts[2].upper(),
            parts[3].upper(),
            parts[4].upper(),
        )
    )


def _strict_entry_theses(path: Path) -> list[dict[str, Any]]:
    """Read one stable, bounded JSONL snapshot and reject malformed rows."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("entry thesis source must be a regular file")
    with resolved.open("rb") as handle:
        descriptor = handle.fileno()
        before_stat = os.fstat(descriptor)
        if before_stat.st_size > MAX_ENTRY_THESIS_BYTES:
            raise ValueError("entry thesis source exceeds the read-only size bound")
        raw = handle.read(MAX_ENTRY_THESIS_BYTES + 1)
        after_stat = os.fstat(descriptor)
    if len(raw) > MAX_ENTRY_THESIS_BYTES:
        raise ValueError("entry thesis source exceeds the read-only size bound")
    if (
        before_stat.st_size != after_stat.st_size
        or before_stat.st_mtime_ns != after_stat.st_mtime_ns
        or len(raw) != before_stat.st_size
    ):
        raise ValueError("entry thesis source changed during the read")
    if raw and not raw.endswith(b"\n"):
        raise ValueError("entry thesis source ends with a partial JSONL row")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("entry thesis source is not UTF-8") from exc

    records: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_non_finite_json_number,
            )
        except (json.JSONDecodeError, _DuplicateJsonKey, ValueError) as exc:
            raise ValueError(f"entry thesis row {index} is invalid") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"entry thesis row {index} must be an object")
        records.append(payload)
    return records


def _review_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    review = entry.get("bot_tuning_review")
    if not isinstance(review, dict):
        review = entry.get("review")
    return review if isinstance(review, dict) else {}


def _review_contract(entry: dict[str, Any]) -> tuple[_ReviewContract | None, str | None]:
    work_order_id = str(entry.get("work_order_id") or "").strip()
    if not work_order_id:
        return None, "WORK_ORDER_ID_MISSING"
    if str(entry.get("status") or "").upper() not in _PENDING_STATUSES:
        return None, "WORK_ORDER_NOT_PENDING"
    if entry.get("consumed_at_utc") not in (None, ""):
        return None, "WORK_ORDER_ALREADY_CONSUMED"
    if (
        entry.get("live_permission_allowed") is not False
        or entry.get("no_direct_oanda") is not True
        or entry.get("preserve_blockers") is not True
    ):
        return None, "WORK_ORDER_SAFETY_BOUNDARY_INVALID"

    observation_id = str(
        entry.get("latest_observation_id")
        or entry.get("observation_id")
        or entry.get("event_fingerprint")
        or ""
    ).strip()
    reviewed_observation_id = str(
        entry.get("latest_reviewed_observation_id") or ""
    ).strip()
    validation = entry.get("bot_tuning_review_validation")
    if (
        not observation_id
        or reviewed_observation_id != observation_id
        or not isinstance(validation, dict)
        or str(validation.get("status") or "").upper() != "VALID"
        or entry.get("review_current", True) is False
    ):
        return None, "CURRENT_VALID_REVIEW_REQUIRED"

    review = _review_from_entry(entry)
    if str(review.get("review_status") or "").upper() != "NO_CHANGE_INSUFFICIENT_EVIDENCE":
        return None, "NO_CHANGE_ACQUISITION_REVIEW_REQUIRED"
    if (
        review.get("live_permission_allowed") is not False
        or review.get("no_direct_oanda") is not True
        or review.get("preserve_blockers") is not True
        or review.get("proposed_adjustments") != []
    ):
        return None, "REVIEW_SAFETY_BOUNDARY_INVALID"

    acquisition = review.get("evidence_acquisition")
    if not isinstance(acquisition, dict):
        return None, "EVIDENCE_ACQUISITION_MISSING"
    if str(acquisition.get("action_kind") or "").upper() != "ADD_PREENTRY_SIGNAL_LOG":
        return None, "ACQUISITION_KIND_UNSUPPORTED"
    if str(acquisition.get("source_ref") or "") != "data/entry_thesis_ledger.jsonl":
        return None, "ACQUISITION_SOURCE_UNSUPPORTED"
    required_count = acquisition.get("required_new_samples")
    if (
        isinstance(required_count, bool)
        or not isinstance(required_count, int)
        or not 1 <= required_count <= 1000
    ):
        return None, "ACQUISITION_SAMPLE_COUNT_INVALID"

    affected_pairs = review.get("affected_pairs")
    affected_families = review.get("affected_bot_families")
    if (
        not isinstance(affected_pairs, list)
        or len(affected_pairs) != 1
        or not isinstance(affected_families, list)
        or len(affected_families) != 1
    ):
        return None, "PAIR_FAMILY_SELECTOR_INVALID"
    pair = str(affected_pairs[0] or "").strip().upper()
    family = str(affected_families[0] or "").strip().lower()
    if not pair or family not in set(_FAMILY_BY_METHOD.values()):
        return None, "PAIR_FAMILY_SELECTOR_UNSUPPORTED"

    selected_event = entry.get("selected_event")
    if not isinstance(selected_event, dict):
        selected_event = {}
    selected_pair = str(selected_event.get("pair") or "").strip().upper()
    if selected_pair and selected_pair != pair:
        return None, "SELECTED_EVENT_PAIR_MISMATCH"
    event_type = str(selected_event.get("event_type") or "").strip().upper()
    details = selected_event.get("details")
    details = details if isinstance(details, dict) else {}
    raw_lane = selected_event.get("lane_id") or details.get("lane_id")
    exact_lane_id = _normalize_lane(raw_lane) if raw_lane else None
    if raw_lane and exact_lane_id is None:
        return None, "SELECTED_EVENT_EXACT_LANE_INVALID"
    if exact_lane_id:
        lane_parts = exact_lane_id.split(":")
        if (
            lane_parts[1] != pair
            or _FAMILY_BY_METHOD.get(lane_parts[3]) != family
        ):
            return None, "SELECTED_EVENT_EXACT_LANE_MISMATCH"

    boundary_text = str(entry.get("structured_review_completed_at_utc") or "").strip()
    try:
        boundary_key = _instant_key(boundary_text)
    except ValueError:
        return None, "STRUCTURED_REVIEW_BOUNDARY_INVALID"
    return (
        _ReviewContract(
            work_order_id=work_order_id,
            observation_id=observation_id,
            boundary_text=boundary_text,
            boundary_key=boundary_key,
            pair=pair,
            family=family,
            event_type=event_type,
            required_count=required_count,
            selector_mode="EXACT_LANE" if exact_lane_id else "PAIR_FAMILY",
            exact_lane_id=exact_lane_id,
        ),
        None,
    )


def _entry_sort_key(entry: AttributedEntry) -> tuple[datetime, int, int]:
    timestamp = entry.broker_entry_ts_utc or entry.entry_ts_utc
    second, nanos = _instant_key(timestamp)
    return second, nanos, int(entry.ledger_rowid)


def _entry_matches(entry: AttributedEntry, contract: _ReviewContract) -> bool:
    try:
        entry_key = _entry_sort_key(entry)[:2]
    except (TypeError, ValueError):
        return False
    canonical_lane = _normalize_lane(entry.canonical_lane_id)
    if (
        entry.broker_time_consistent is not True
        or not canonical_lane
        or str(entry.pair or "").upper() != contract.pair
        or entry_key <= contract.boundary_key
    ):
        return False
    if contract.exact_lane_id:
        return canonical_lane == contract.exact_lane_id
    return _FAMILY_BY_METHOD.get(str(entry.method or "").upper()) == contract.family


def _thesis_index(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for payload in records:
        trade_id = str(payload.get("trade_id") or "").strip()
        if trade_id:
            result.setdefault(trade_id, []).append(payload)
    return result


def _signal_state_defects(
    *,
    state: object,
    event_type: str,
    family: str,
) -> list[str]:
    if not isinstance(state, dict) or not state:
        return ["GUARDIAN_TUNING_SIGNAL_STATE_MISSING"]
    if event_type == "TECHNICAL_STATE_CHANGE":
        defects: list[str] = []
        if not (_TECHNICAL_STATE_KEYS & set(state)):
            defects.append("TECHNICAL_REGIME_STATE_MISSING")
        family_keys = _TECHNICAL_FAMILY_KEYS.get(family, frozenset())
        if not (family_keys & set(state)):
            defects.append(f"{family.upper()}_FAMILY_SIGNAL_MISSING")
        return defects
    if event_type != "FAILED_ACCEPTANCE":
        return []
    defects: list[str] = []
    if state.get("failed_acceptance") is not True:
        defects.append("FAILED_ACCEPTANCE_PREDICATE_MISSING")
    acceptance_zone = state.get("acceptance_zone")
    if (
        acceptance_zone is None
        or acceptance_zone == ""
        or isinstance(acceptance_zone, bool)
        or (
            isinstance(acceptance_zone, (int, float))
            and not math.isfinite(float(acceptance_zone))
        )
    ):
        defects.append("FAILED_ACCEPTANCE_ZONE_MISSING")
    return defects


def _thesis_defects(
    *,
    entry: AttributedEntry,
    matches: Sequence[dict[str, Any]],
    contract: _ReviewContract,
) -> list[str]:
    if not matches:
        return ["ENTRY_THESIS_MISSING"]
    if len(matches) != 1:
        return ["ENTRY_THESIS_NOT_UNIQUE"]
    thesis = matches[0]
    evidence = thesis.get("context_evidence")
    if not isinstance(evidence, dict):
        return ["ENTRY_THESIS_CONTEXT_MISSING"]

    defects: list[str] = []
    if str(thesis.get("pair") or "").upper() != str(entry.pair or "").upper():
        defects.append("ENTRY_THESIS_PAIR_MISMATCH")
    if str(thesis.get("side") or "").upper() != str(entry.side or "").upper():
        defects.append("ENTRY_THESIS_SIDE_MISMATCH")
    if str(evidence.get("order_id") or "") != str(entry.order_id or ""):
        defects.append("ENTRY_THESIS_ORDER_ID_MISMATCH")
    if _normalize_lane(evidence.get("lane_id")) != _normalize_lane(
        entry.canonical_lane_id
    ):
        defects.append("ENTRY_THESIS_EXACT_LANE_MISMATCH")
    try:
        thesis_at = _instant_key(thesis.get("timestamp_utc"))
        entry_at = _instant_key(entry.broker_entry_ts_utc or entry.entry_ts_utc)
    except ValueError:
        defects.append("ENTRY_THESIS_TIMESTAMP_INVALID")
    else:
        if thesis_at != entry_at:
            defects.append("ENTRY_THESIS_TIMESTAMP_MISMATCH")
    defects.extend(
        _signal_state_defects(
            state=evidence.get("guardian_tuning_signal_state"),
            event_type=contract.event_type,
            family=contract.family,
        )
    )
    return list(dict.fromkeys(defects))


def _canonical_outcome_matches(
    *,
    entry: AttributedEntry,
    outcomes: Sequence[RealizedOutcome],
) -> tuple[bool, list[str]]:
    matches = [row for row in outcomes if str(row.trade_id) == str(entry.trade_id)]
    if not matches:
        return False, []
    if len(matches) != 1:
        return False, ["CANONICAL_OUTCOME_NOT_UNIQUE"]
    outcome = matches[0]
    outcome_lane = _normalize_lane(outcome.lane_id)
    if outcome_lane is None:
        lane_parts = [part.strip() for part in str(outcome.lane_id or "").split(":")]
        if len(lane_parts) == 4:
            outcome_lane = _normalize_lane(
                ":".join((*lane_parts, str(outcome.entry_vehicle or "")))
            )
    if (
        str(outcome.pair or "").upper() != str(entry.pair or "").upper()
        or str(outcome.side or "").upper() != str(entry.side or "").upper()
        or str(outcome.method or "").upper() != str(entry.method or "").upper()
        or outcome_lane != _normalize_lane(entry.canonical_lane_id)
        or outcome.entry_truth_consistent is not True
        or outcome.broker_time_consistent is not True
    ):
        return False, ["CANONICAL_OUTCOME_IDENTITY_MISMATCH"]
    return True, []


def _source_defect_progress(
    contract: _ReviewContract,
    *,
    source_issue_codes: Sequence[str],
) -> dict[str, Any]:
    return {
        "work_order_id": contract.work_order_id,
        "observation_id": contract.observation_id,
        "status": "ACQUISITION_SOURCE_DEFECT",
        "boundary_utc": contract.boundary_text,
        "selector": {
            "mode": contract.selector_mode,
            "pair": contract.pair,
            "bot_family": contract.family,
            "lane_id": contract.exact_lane_id,
        },
        "required_count": contract.required_count,
        "entry_count": 0,
        "preentry_complete_count": 0,
        "resolved_count": 0,
        "complete_count": 0,
        "remaining_entry_count": contract.required_count,
        "remaining_preentry_count": contract.required_count,
        "remaining_resolution_count": contract.required_count,
        "remaining_complete_count": contract.required_count,
        "first_trade_ids": [],
        "signal_defects": [],
        "outcome_defects": [],
        "source_issue_codes": list(dict.fromkeys(source_issue_codes)),
    }


def _progress_for_contract(
    *,
    contract: _ReviewContract,
    entries: Sequence[AttributedEntry],
    outcomes: Sequence[RealizedOutcome],
    theses_by_trade: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    try:
        matching_entries = sorted(
            (entry for entry in entries if _entry_matches(entry, contract)),
            key=_entry_sort_key,
        )
    except (TypeError, ValueError):
        return _source_defect_progress(
            contract,
            source_issue_codes=["CANONICAL_ENTRY_TIMESTAMP_INVALID"],
        )
    first_entries = matching_entries[: contract.required_count]

    signal_defects: list[dict[str, Any]] = []
    outcome_defects: list[dict[str, Any]] = []
    preentry_complete_ids: set[str] = set()
    resolved_ids: set[str] = set()
    for entry in first_entries:
        trade_id = str(entry.trade_id)
        thesis_issues = _thesis_defects(
            entry=entry,
            matches=theses_by_trade.get(trade_id, []),
            contract=contract,
        )
        if thesis_issues:
            signal_defects.append({"trade_id": trade_id, "codes": thesis_issues})
        else:
            preentry_complete_ids.add(trade_id)
        resolved, resolution_issues = _canonical_outcome_matches(
            entry=entry,
            outcomes=outcomes,
        )
        if resolution_issues:
            outcome_defects.append(
                {"trade_id": trade_id, "codes": resolution_issues}
            )
        elif resolved:
            resolved_ids.add(trade_id)

    entry_count = len(first_entries)
    preentry_complete_count = len(preentry_complete_ids)
    resolved_count = len(resolved_ids)
    complete_count = len(preentry_complete_ids & resolved_ids)
    defects_present = bool(signal_defects or outcome_defects)
    if entry_count < contract.required_count:
        status = (
            "COLLECTING_WITH_SIGNAL_DEFECT"
            if defects_present
            else "WAITING_FOR_ENTRIES"
        )
    elif defects_present:
        status = "ACQUISITION_SOURCE_DEFECT"
    elif resolved_count < contract.required_count:
        status = "WAITING_FOR_RESOLUTION"
    else:
        status = "READY_FOR_GPT_REVIEW_UPGRADE"

    return {
        "work_order_id": contract.work_order_id,
        "observation_id": contract.observation_id,
        "status": status,
        "boundary_utc": contract.boundary_text,
        "selector": {
            "mode": contract.selector_mode,
            "pair": contract.pair,
            "bot_family": contract.family,
            "lane_id": contract.exact_lane_id,
        },
        "required_count": contract.required_count,
        "entry_count": entry_count,
        "preentry_complete_count": preentry_complete_count,
        "resolved_count": resolved_count,
        "complete_count": complete_count,
        "remaining_entry_count": max(contract.required_count - entry_count, 0),
        "remaining_preentry_count": max(
            contract.required_count - preentry_complete_count,
            0,
        ),
        "remaining_resolution_count": max(
            contract.required_count - resolved_count,
            0,
        ),
        "remaining_complete_count": max(
            contract.required_count - complete_count,
            0,
        ),
        "first_trade_ids": [str(entry.trade_id) for entry in first_entries],
        "signal_defects": signal_defects,
        "outcome_defects": outcome_defects,
        "source_issue_codes": [],
    }


def _input_work_orders(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        raw = value.get("work_orders")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, dict)]
    return []


def build_guardian_tuning_acquisition_progress(
    work_orders: object,
    *,
    entry_thesis_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    """Report forward evidence progress for supported current queue reviews.

    The paths are required arguments on purpose: callers and tests cannot fall
    through to a repository or live-runtime artifact accidentally.
    """

    contracts: list[_ReviewContract] = []
    unsupported: list[dict[str, Any]] = []
    for entry in _input_work_orders(work_orders):
        contract, reason = _review_contract(entry)
        if contract is None:
            unsupported.append(
                {
                    "work_order_id": str(entry.get("work_order_id") or "") or None,
                    "reason_code": reason or "UNSUPPORTED_REVIEW",
                }
            )
            continue
        contracts.append(contract)

    if not contracts:
        return {
            "schema_version": 1,
            "status": "NO_SUPPORTED_ACQUISITION_WORK",
            "counts": {
                "supported": 0,
                "ready": 0,
                "waiting": 0,
                "defect": 0,
                "unsupported": len(unsupported),
            },
            "work_orders": [],
            "unsupported_work_orders": unsupported,
            "evidence_mode": "FORWARD_FIRST_N_CANONICAL_ENTRIES",
        }

    source_issue_codes: list[str] = []
    try:
        thesis_records = _strict_entry_theses(Path(entry_thesis_path))
    except (OSError, RuntimeError, TypeError, ValueError):
        thesis_records = []
        source_issue_codes.append("ENTRY_THESIS_SOURCE_UNREADABLE")
    try:
        entries = read_attributed_system_entries(Path(ledger_path))
    except (OSError, RuntimeError, TypeError, ValueError):
        entries = None
    if entries is None:
        source_issue_codes.append("CANONICAL_ENTRY_LEDGER_UNREADABLE")
    try:
        outcomes = read_attributed_net_outcomes(Path(ledger_path))
    except (OSError, RuntimeError, TypeError, ValueError):
        outcomes = None
    if outcomes is None:
        source_issue_codes.append("CANONICAL_OUTCOME_LEDGER_UNREADABLE")

    if source_issue_codes:
        progress = [
            _source_defect_progress(
                contract,
                source_issue_codes=source_issue_codes,
            )
            for contract in contracts
        ]
    else:
        theses_by_trade = _thesis_index(thesis_records)
        progress = [
            _progress_for_contract(
                contract=contract,
                entries=entries or [],
                outcomes=outcomes or [],
                theses_by_trade=theses_by_trade,
            )
            for contract in contracts
        ]

    ready_count = sum(
        item["status"] == "READY_FOR_GPT_REVIEW_UPGRADE" for item in progress
    )
    defect_count = sum(
        item["status"]
        in {"COLLECTING_WITH_SIGNAL_DEFECT", "ACQUISITION_SOURCE_DEFECT"}
        for item in progress
    )
    waiting_count = len(progress) - ready_count - defect_count
    overall_status = (
        "SOURCE_DEFECT"
        if defect_count
        else "READY_FOR_GPT_REVIEW_UPGRADE"
        if ready_count == len(progress)
        else "COLLECTING_FORWARD_EVIDENCE"
    )
    return {
        "schema_version": 1,
        "status": overall_status,
        "counts": {
            "supported": len(progress),
            "ready": ready_count,
            "waiting": waiting_count,
            "defect": defect_count,
            "unsupported": len(unsupported),
        },
        "work_orders": progress,
        "unsupported_work_orders": unsupported,
        "evidence_mode": "FORWARD_FIRST_N_CANONICAL_ENTRIES",
    }
