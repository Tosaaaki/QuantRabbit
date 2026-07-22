"""Tamper-evident paper-only evidence loop for autonomous DOJO research.

This module does not decide or execute trades. It supplies the measurement
boundary required before AI narrative or inventory opinions can be evaluated:
cutoff-bound shadow assessments, post-horizon outcome scoring, and an
append-only one-candidate replay lifecycle. Every write validates the complete
existing chain under a file lock. All records remain paper-only with order
authority NONE; no broker client or execution callback is imported here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SHADOW_LEDGER_CONTRACT = "QR_DOJO_AI_SHADOW_LEDGER_V1"
CANDIDATE_LEDGER_CONTRACT = "QR_DOJO_AUTONOMOUS_CANDIDATE_LEDGER_V1"
SHADOW_ASSESSMENT_CONTRACT = "QR_DOJO_AI_SHADOW_ASSESSMENT_V1"
SHADOW_OUTCOME_CONTRACT = "QR_DOJO_AI_SHADOW_OUTCOME_V1"
CANDIDATE_SPEC_CONTRACT = "QR_DOJO_AUTONOMOUS_CANDIDATE_SPEC_V1"
ACTIVE_CHECKPOINT_CONTRACT = "QR_DOJO_AUTONOMOUS_ACTIVE_CANDIDATE_V1"

REGIMES = frozenset({"TREND", "RANGE", "SQUEEZE", "EVENT", "UNCLEAR"})
SUPERVISION = frozenset({"GO", "CAUTION", "STOP"})
POSITION_STATES = frozenset({"ALIVE", "WOUNDED", "INVALIDATED", "UNCLEAR"})
INVENTORY_STATES = frozenset(
    {"BALANCED", "CONCENTRATED", "TRAPPED", "CAPITAL_LOCKED"}
)
SHADOW_ACTIONS = frozenset(
    {"OBSERVE_HOLD", "NO_NEW_ENTRY_TEST", "EXIT_TEST_CANDIDATE"}
)
CANDIDATE_FAMILIES = frozenset(
    {
        "CONDITIONAL_EXIT",
        "ENTRY_REGIME",
        "INVENTORY_RELEASE",
        "COST_EXECUTION",
        "PAIR_ADAPTATION",
        "NEWS_CONDITIONED_SHADOW",
    }
)
DEATH_CODES = frozenset(
    {
        "COST",
        "DIRECTION",
        "EXIT_TIMING",
        "REGIME_MISMATCH",
        "INVENTORY",
        "OVERFIT",
        "MEASUREMENT",
        "RISK",
    }
)
CANDIDATE_EVENTS = frozenset(
    {
        "SYSTEM_BOOTSTRAPPED",
        "CANDIDATE_PREREGISTERED",
        "REPLAY_STARTED",
        "REPLAY_REJECTED",
        "REPLAY_PASSED",
        "PAPER_ELIGIBLE",
    }
)
TERMINAL_CANDIDATE_EVENTS = frozenset({"REPLAY_REJECTED", "PAPER_ELIGIBLE"})
_SHA_LENGTH = 64
MAX_ASSESSMENT_RECORD_LAG_SECONDS = 300


class DojoAutonomousEvidenceError(ValueError):
    """Point-in-time evidence or an append-only chain is invalid."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _utc(value: Any, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DojoAutonomousEvidenceError(
                f"{label} must be an ISO timestamp"
            ) from exc
    else:
        raise DojoAutonomousEvidenceError(f"{label} must be a timestamp")
    if parsed.tzinfo is None:
        raise DojoAutonomousEvidenceError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sha(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != _SHA_LENGTH or any(
        ch not in "0123456789abcdef" for ch in text
    ):
        raise DojoAutonomousEvidenceError(f"{label} must be a lowercase sha256")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoAutonomousEvidenceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DojoAutonomousEvidenceError(f"{label} must be finite")
    return number


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise DojoAutonomousEvidenceError(f"{label} is required")
    return text


def _paper_guard(value: Mapping[str, Any], label: str) -> None:
    if (
        value.get("order_authority") != "NONE"
        or value.get("paper_only") is not True
        or value.get("live_permission") is not False
    ):
        raise DojoAutonomousEvidenceError(
            f"{label} must be paper-only authority NONE"
        )


def build_shadow_assessment(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and seal one AI assessment using only cutoff-bound evidence."""

    if not isinstance(payload, Mapping):
        raise DojoAutonomousEvidenceError("assessment payload must be an object")
    body = dict(payload)
    body.pop("assessment_sha256", None)
    body.pop("assessment_id", None)
    if body.get("contract") != SHADOW_ASSESSMENT_CONTRACT:
        raise DojoAutonomousEvidenceError("assessment contract is invalid")
    _paper_guard(body, "assessment")
    as_of = _utc(body.get("as_of_utc"), "as_of_utc")
    horizon = _utc(body.get("horizon_end_utc"), "horizon_end_utc")
    if horizon <= as_of:
        raise DojoAutonomousEvidenceError(
            "horizon_end_utc must be after as_of_utc"
        )
    for field in ("ledger_sha256", "state_sha256", "snapshot_sha256"):
        _sha(body.get(field), field)
    pair = _required_text(body.get("pair"), "pair").upper()
    family = _required_text(body.get("strategy_family"), "strategy_family").upper()
    body["pair"] = pair
    body["strategy_family"] = family
    if body.get("regime") not in REGIMES:
        raise DojoAutonomousEvidenceError("regime is invalid")
    if body.get("supervision") not in SUPERVISION:
        raise DojoAutonomousEvidenceError("supervision is invalid")
    confidence = _finite(body.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise DojoAutonomousEvidenceError(
            "confidence must be between zero and one"
        )
    facts = body.get("facts")
    if not isinstance(facts, list) or not 1 <= len(facts) <= 3:
        raise DojoAutonomousEvidenceError(
            "facts must contain one to three items"
        )
    body["facts"] = [_required_text(item, "fact") for item in facts]
    for field in (
        "primary_path",
        "alternative_path",
        "falsifier",
        "habitat_reason",
    ):
        body[field] = _required_text(body.get(field), field)

    quote = body.get("quote")
    if not isinstance(quote, Mapping) or str(quote.get("pair") or "").upper() != pair:
        raise DojoAutonomousEvidenceError("quote must bind the assessment pair")
    quote_ts = _utc(quote.get("ts_utc"), "quote.ts_utc")
    if quote_ts > as_of:
        raise DojoAutonomousEvidenceError("quote timestamp is after as_of_utc")
    bid = _finite(quote.get("bid"), "quote.bid")
    ask = _finite(quote.get("ask"), "quote.ask")
    if bid <= 0.0 or ask < bid:
        raise DojoAutonomousEvidenceError("quote bid/ask geometry is invalid")

    watermarks = body.get("source_watermarks")
    if not isinstance(watermarks, list) or not watermarks:
        raise DojoAutonomousEvidenceError("source_watermarks must not be empty")
    normalized_watermarks: list[dict[str, Any]] = []
    for index, item in enumerate(watermarks):
        if not isinstance(item, Mapping):
            raise DojoAutonomousEvidenceError(
                f"source watermark {index} is invalid"
            )
        observed = _utc(
            item.get("observed_through_utc"), f"watermark {index} time"
        )
        if observed > as_of:
            raise DojoAutonomousEvidenceError(
                "source watermark is after as_of_utc"
            )
        normalized_watermarks.append(
            {
                "kind": _required_text(
                    item.get("kind"), f"watermark {index} kind"
                ),
                "observed_through_utc": observed.isoformat(),
                "sha256": _sha(
                    item.get("sha256"), f"watermark {index} sha256"
                ),
            }
        )
    body["source_watermarks"] = normalized_watermarks

    positions = body.get("positions")
    if not isinstance(positions, list):
        raise DojoAutonomousEvidenceError("positions must be a list")
    seen_positions: set[str] = set()
    normalized_positions: list[dict[str, Any]] = []
    for index, item in enumerate(positions):
        if not isinstance(item, Mapping):
            raise DojoAutonomousEvidenceError(f"position {index} is invalid")
        position = dict(item)
        identity = _required_text(position.get("position_id"), "position_id")
        if identity in seen_positions:
            raise DojoAutonomousEvidenceError("position identity is duplicated")
        seen_positions.add(identity)
        if position.get("thesis") not in POSITION_STATES:
            raise DojoAutonomousEvidenceError("position thesis is invalid")
        if position.get("inventory") not in INVENTORY_STATES:
            raise DojoAutonomousEvidenceError(
                "position inventory state is invalid"
            )
        if position.get("shadow_action") not in SHADOW_ACTIONS:
            raise DojoAutonomousEvidenceError("position shadow action is invalid")
        _sha(position.get("entry_context_sha256"), "entry_context_sha256")
        opened = _utc(position.get("opened_at_utc"), "opened_at_utc")
        if opened > as_of:
            raise DojoAutonomousEvidenceError(
                "position opened after as_of_utc"
            )
        for field in (
            "units",
            "entry_price",
            "executable_mark",
            "unrealized_pnl_jpy",
            "tp_progress",
            "ceiling_remaining_minutes",
            "margin_usage",
            "capital_lock_jpy",
        ):
            _finite(position.get(field), field)
        position["opened_at_utc"] = opened.isoformat()
        normalized_positions.append(position)
    body["positions"] = normalized_positions
    body["as_of_utc"] = as_of.isoformat()
    body["horizon_end_utc"] = horizon.isoformat()
    assessment_id = canonical_sha256(body)
    sealed = {**body, "assessment_id": assessment_id}
    return {**sealed, "assessment_sha256": canonical_sha256(sealed)}


def build_shadow_outcome(
    payload: Mapping[str, Any],
    *,
    assessment: Mapping[str, Any],
    recorded_at_utc: datetime | str,
) -> dict[str, Any]:
    """Seal an outcome only after the assessment horizon or its settlement."""

    if not isinstance(payload, Mapping):
        raise DojoAutonomousEvidenceError("outcome payload must be an object")
    if build_shadow_assessment(assessment) != dict(assessment):
        raise DojoAutonomousEvidenceError("bound assessment is not canonical")
    body = dict(payload)
    body.pop("outcome_sha256", None)
    body.pop("outcome_id", None)
    if body.get("contract") != SHADOW_OUTCOME_CONTRACT:
        raise DojoAutonomousEvidenceError("outcome contract is invalid")
    _paper_guard(body, "outcome")
    if body.get("assessment_id") != assessment.get("assessment_id"):
        raise DojoAutonomousEvidenceError(
            "outcome assessment binding is invalid"
        )
    recorded = _utc(recorded_at_utc, "recorded_at_utc")
    horizon = _utc(assessment.get("horizon_end_utc"), "assessment horizon")
    as_of = _utc(assessment.get("as_of_utc"), "assessment cutoff")
    settled_at_raw = body.get("settled_at_utc")
    settled_at = (
        _utc(settled_at_raw, "settled_at_utc") if settled_at_raw else None
    )
    if recorded < horizon and (settled_at is None or settled_at > recorded):
        raise DojoAutonomousEvidenceError("outcome is not mature")
    if settled_at is not None and settled_at <= as_of:
        raise DojoAutonomousEvidenceError(
            "settlement must be after the assessment cutoff"
        )
    observed = _utc(body.get("observed_through_utc"), "observed_through_utc")
    if observed < as_of or observed > recorded:
        raise DojoAutonomousEvidenceError(
            "outcome watermark is outside the scoring cutoff"
        )
    for field in (
        "realized_pnl_jpy",
        "mfe_pips",
        "mae_pips",
        "actual_exit_price",
        "counterfactual_exit_price",
        "counterfactual_delta_jpy",
    ):
        _finite(body.get(field), field)
    if not isinstance(body.get("regime_correct"), bool):
        raise DojoAutonomousEvidenceError("regime_correct must be boolean")
    body["observed_through_utc"] = observed.isoformat()
    body["settled_at_utc"] = settled_at.isoformat() if settled_at else None
    body["scored_at_utc"] = recorded.isoformat()
    outcome_id = canonical_sha256(body)
    sealed = {**body, "outcome_id": outcome_id}
    return {**sealed, "outcome_sha256": canonical_sha256(sealed)}


def build_candidate_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a one-logical-change replay proposal before selection begins."""

    if not isinstance(payload, Mapping):
        raise DojoAutonomousEvidenceError("candidate spec must be an object")
    body = dict(payload)
    body.pop("candidate_id", None)
    body.pop("spec_sha256", None)
    if body.get("contract") != CANDIDATE_SPEC_CONTRACT:
        raise DojoAutonomousEvidenceError("candidate spec contract is invalid")
    _paper_guard(body, "candidate spec")
    family = str(body.get("family") or "").upper()
    if family not in CANDIDATE_FAMILIES:
        raise DojoAutonomousEvidenceError("candidate family is invalid")
    body["family"] = family
    for field in (
        "hypothesis",
        "causal_narrative",
        "expected_mechanism",
        "falsifier",
        "affected_pair",
        "affected_strategy",
        "evidence_cohort",
    ):
        body[field] = _required_text(body.get(field), field)
    rule = body.get("changed_rule")
    if not isinstance(rule, Mapping) or set(rule) != {
        "name",
        "baseline",
        "candidate",
    }:
        raise DojoAutonomousEvidenceError(
            "changed_rule must describe exactly one change"
        )
    if rule.get("baseline") == rule.get("candidate"):
        raise DojoAutonomousEvidenceError(
            "candidate value must differ from baseline"
        )
    controls = body.get("unchanged_controls")
    if not isinstance(controls, list) or not controls:
        raise DojoAutonomousEvidenceError(
            "unchanged_controls must not be empty"
        )
    evidence = body.get("evidence_sha256s")
    if not isinstance(evidence, list) or not evidence:
        raise DojoAutonomousEvidenceError("evidence_sha256s must not be empty")
    body["evidence_sha256s"] = [
        _sha(item, "evidence sha256") for item in evidence
    ]
    windows = body.get("windows")
    if not isinstance(windows, Mapping) or set(windows) != {
        "TRAIN",
        "VAL",
        "S5",
    }:
        raise DojoAutonomousEvidenceError(
            "windows must contain TRAIN, VAL, and S5"
        )
    normalized_windows: dict[str, Any] = {}
    last_end: datetime | None = None
    for name in ("TRAIN", "VAL", "S5"):
        item = windows[name]
        if not isinstance(item, Mapping):
            raise DojoAutonomousEvidenceError(f"{name} window is invalid")
        start = _utc(item.get("from_utc"), f"{name}.from_utc")
        end = _utc(item.get("to_utc"), f"{name}.to_utc")
        if start >= end or (last_end is not None and start < last_end):
            raise DojoAutonomousEvidenceError(
                "candidate windows overlap or are not positive"
            )
        last_end = end
        normalized_windows[name] = {
            "from_utc": start.isoformat(),
            "to_utc": end.isoformat(),
            "source_sha256": _sha(
                item.get("source_sha256"), f"{name}.source_sha256"
            ),
        }
    body["windows"] = normalized_windows
    costs = body.get("costs")
    if not isinstance(costs, Mapping) or set(costs) != {"BASE", "STRESS"}:
        raise DojoAutonomousEvidenceError("BASE and STRESS costs are required")
    for name in ("BASE", "STRESS"):
        if not isinstance(costs[name], Mapping):
            raise DojoAutonomousEvidenceError(f"{name} costs are invalid")
        for field in (
            "slippage_pips_per_fill",
            "financing_pips_per_day",
        ):
            if _finite(costs[name].get(field), f"{name}.{field}") < 0.0:
                raise DojoAutonomousEvidenceError(
                    "cost inputs cannot be negative"
                )
    if body.get("intrabar_paths") != ["OHLC", "OLHC"]:
        raise DojoAutonomousEvidenceError("both intrabar paths must be fixed")
    if body.get("end_of_replay_forced_close_benefit") is not False:
        raise DojoAutonomousEvidenceError(
            "forced-close benefit must be disabled"
        )
    gates = body.get("risk_gates")
    if (
        not isinstance(gates, Mapping)
        or float(gates.get("min_independent_stress_pf", 0.0)) < 1.25
    ):
        raise DojoAutonomousEvidenceError(
            "risk gates must require stress PF >= 1.25"
        )
    if body.get("death_codes") != sorted(DEATH_CODES):
        raise DojoAutonomousEvidenceError(
            "candidate must freeze the complete rejection taxonomy"
        )
    candidate_id = canonical_sha256(body)
    sealed = {**body, "candidate_id": candidate_id}
    return {**sealed, "spec_sha256": canonical_sha256(sealed)}


def _read_ledger(handle: Any, contract: str) -> list[dict[str, Any]]:
    handle.seek(0)
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    previous_time: datetime | None = None
    for line_number, raw in enumerate(handle, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DojoAutonomousEvidenceError(
                f"ledger JSON invalid at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise DojoAutonomousEvidenceError(
                f"ledger row is not an object at line {line_number}"
            )
        body = {key: value for key, value in row.items() if key != "event_sha256"}
        if (
            row.get("contract") != contract
            or row.get("sequence") != len(rows) + 1
            or row.get("previous_event_sha256") != previous
            or row.get("event_sha256") != canonical_sha256(body)
        ):
            raise DojoAutonomousEvidenceError(
                f"ledger chain invalid at line {line_number}"
            )
        _paper_guard(row, f"ledger row {line_number}")
        recorded = _utc(row.get("recorded_at_utc"), "recorded_at_utc")
        if previous_time is not None and recorded < previous_time:
            raise DojoAutonomousEvidenceError("ledger time regressed")
        if row.get("payload_sha256") != canonical_sha256(row.get("payload")):
            raise DojoAutonomousEvidenceError(
                "ledger payload digest is invalid"
            )
        previous = row["event_sha256"]
        previous_time = recorded
        rows.append(row)
    return rows


def validate_event_ledger(path: Path, *, contract: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "rows": 0, "tip_sha256": None}
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            rows = _read_ledger(handle, contract)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {
        "status": "VALID",
        "rows": len(rows),
        "tip_sha256": rows[-1]["event_sha256"] if rows else None,
    }


def _validate_shadow_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    assessments: dict[str, Mapping[str, Any]] = {}
    outcomes: set[str] = set()
    for index, row in enumerate(rows, 1):
        event_type = row.get("event_type")
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            raise DojoAutonomousEvidenceError(
                f"shadow payload invalid at row {index}"
            )
        if event_type == "ASSESSMENT_RECORDED":
            rebuilt = build_shadow_assessment(payload)
            if rebuilt != dict(payload):
                raise DojoAutonomousEvidenceError(
                    f"shadow assessment is not canonical at row {index}"
                )
            assessment_id = str(payload["assessment_id"])
            recorded = _utc(row.get("recorded_at_utc"), "recorded_at_utc")
            as_of = _utc(payload.get("as_of_utc"), "as_of_utc")
            lag = (recorded - as_of).total_seconds()
            if not 0 <= lag <= MAX_ASSESSMENT_RECORD_LAG_SECONDS:
                raise DojoAutonomousEvidenceError(
                    "shadow assessment record lag permits hindsight"
                )
            if row.get("identity") != assessment_id:
                raise DojoAutonomousEvidenceError(
                    "shadow assessment ledger identity is invalid"
                )
            if assessment_id in assessments:
                raise DojoAutonomousEvidenceError(
                    "shadow assessment identity is duplicated"
                )
            assessments[assessment_id] = payload
        elif event_type == "OUTCOME_RECORDED":
            assessment_id = str(payload.get("assessment_id") or "")
            assessment = assessments.get(assessment_id)
            if assessment is None or assessment_id in outcomes:
                raise DojoAutonomousEvidenceError(
                    "shadow outcome is missing or duplicates its assessment"
                )
            rebuilt = build_shadow_outcome(
                payload,
                assessment=assessment,
                recorded_at_utc=row.get("recorded_at_utc"),
            )
            if rebuilt != dict(payload):
                raise DojoAutonomousEvidenceError(
                    f"shadow outcome is not canonical at row {index}"
                )
            if row.get("identity") != payload.get("outcome_id"):
                raise DojoAutonomousEvidenceError(
                    "shadow outcome ledger identity is invalid"
                )
            outcomes.add(assessment_id)
        else:
            raise DojoAutonomousEvidenceError(
                f"shadow event type invalid at row {index}"
            )


def validate_shadow_ledger(path: Path) -> dict[str, Any]:
    result = validate_event_ledger(path, contract=SHADOW_LEDGER_CONTRACT)
    if result["status"] == "MISSING":
        return result
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            rows = _read_ledger(handle, SHADOW_LEDGER_CONTRACT)
            _validate_shadow_rows(rows)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return result


def _append_event(
    path: Path,
    *,
    contract: str,
    event_type: str,
    payload: Mapping[str, Any],
    recorded_at_utc: datetime | str,
    identity: str,
) -> tuple[dict[str, Any], bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    recorded = _utc(recorded_at_utc, "recorded_at_utc")
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            rows = _read_ledger(handle, contract)
            for row in rows:
                if row.get("identity") == identity:
                    if row.get("payload_sha256") != canonical_sha256(payload):
                        raise DojoAutonomousEvidenceError(
                            "ledger identity collision"
                        )
                    return row, False
            previous = rows[-1]["event_sha256"] if rows else None
            if rows and recorded < _utc(
                rows[-1].get("recorded_at_utc"), "previous recorded_at_utc"
            ):
                raise DojoAutonomousEvidenceError(
                    "new ledger event time regressed"
                )
            body = {
                "contract": contract,
                "sequence": len(rows) + 1,
                "event_type": event_type,
                "identity": identity,
                "recorded_at_utc": recorded.isoformat(),
                "previous_event_sha256": previous,
                "payload_sha256": canonical_sha256(payload),
                "payload": dict(payload),
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
            row = {**body, "event_sha256": canonical_sha256(body)}
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return row, True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_shadow_assessment(
    path: Path,
    payload: Mapping[str, Any],
    *,
    recorded_at_utc: datetime | str,
) -> tuple[dict[str, Any], bool]:
    assessment = build_shadow_assessment(payload)
    recorded = _utc(recorded_at_utc, "recorded_at_utc")
    as_of = _utc(assessment["as_of_utc"], "as_of_utc")
    lag_seconds = (recorded - as_of).total_seconds()
    if lag_seconds < 0:
        raise DojoAutonomousEvidenceError(
            "assessment cannot be recorded before its cutoff"
        )
    if lag_seconds > MAX_ASSESSMENT_RECORD_LAG_SECONDS:
        raise DojoAutonomousEvidenceError(
            "assessment record lag permits hindsight"
        )
    return _append_event(
        path,
        contract=SHADOW_LEDGER_CONTRACT,
        event_type="ASSESSMENT_RECORDED",
        payload=assessment,
        recorded_at_utc=recorded,
        identity=assessment["assessment_id"],
    )


def append_shadow_outcome(
    path: Path,
    payload: Mapping[str, Any],
    *,
    recorded_at_utc: datetime | str,
) -> tuple[dict[str, Any], bool]:
    lock_path = path.with_name(f".{path.name}.outcome.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lifecycle_lock:
        fcntl.flock(lifecycle_lock.fileno(), fcntl.LOCK_EX)
        try:
            if not path.exists():
                raise DojoAutonomousEvidenceError("shadow ledger is missing")
            with path.open("r", encoding="utf-8") as handle:
                rows = _read_ledger(handle, SHADOW_LEDGER_CONTRACT)
            assessment_id = str(payload.get("assessment_id") or "")
            assessments = {
                row["payload"]["assessment_id"]: row["payload"]
                for row in rows
                if row.get("event_type") == "ASSESSMENT_RECORDED"
            }
            if assessment_id not in assessments:
                raise DojoAutonomousEvidenceError("bound assessment is missing")
            if any(
                row.get("event_type") == "OUTCOME_RECORDED"
                and row.get("payload", {}).get("assessment_id") == assessment_id
                for row in rows
            ):
                raise DojoAutonomousEvidenceError(
                    "assessment outcome already exists"
                )
            outcome = build_shadow_outcome(
                payload,
                assessment=assessments[assessment_id],
                recorded_at_utc=recorded_at_utc,
            )
            return _append_event(
                path,
                contract=SHADOW_LEDGER_CONTRACT,
                event_type="OUTCOME_RECORDED",
                payload=outcome,
                recorded_at_utc=recorded_at_utc,
                identity=outcome["outcome_id"],
            )
        finally:
            fcntl.flock(lifecycle_lock.fileno(), fcntl.LOCK_UN)


def _candidate_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    active: dict[str, Any] | None = None
    for row in rows:
        event = row.get("event_type")
        payload = row.get("payload") or {}
        if event == "SYSTEM_BOOTSTRAPPED":
            continue
        candidate_id = str(payload.get("candidate_id") or "")
        if event == "CANDIDATE_PREREGISTERED":
            if active is not None:
                raise DojoAutonomousEvidenceError(
                    "multiple active candidates detected"
                )
            active = {"candidate_id": candidate_id, "status": "PREREGISTERED"}
        else:
            if active is None or active["candidate_id"] != candidate_id:
                raise DojoAutonomousEvidenceError(
                    "candidate lifecycle transition is unbound"
                )
            allowed = {
                "PREREGISTERED": {"REPLAY_STARTED", "REPLAY_REJECTED"},
                "STARTED": {"REPLAY_REJECTED", "REPLAY_PASSED"},
                "PASSED": {"PAPER_ELIGIBLE"},
            }.get(active["status"], set())
            if event not in allowed:
                raise DojoAutonomousEvidenceError(
                    "candidate lifecycle transition is invalid"
                )
            active["status"] = {
                "REPLAY_STARTED": "STARTED",
                "REPLAY_PASSED": "PASSED",
                "REPLAY_REJECTED": "REJECTED",
                "PAPER_ELIGIBLE": "PAPER_ELIGIBLE",
            }[event]
            if event in TERMINAL_CANDIDATE_EVENTS:
                active = None
    return active


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_active_checkpoint(
    root: Path, active: Mapping[str, Any] | None, tip: str
) -> None:
    body = {
        "contract": ACTIVE_CHECKPOINT_CONTRACT,
        "active_candidate": dict(active) if active else None,
        "candidate_ledger_tip_sha256": tip,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    _atomic_json(
        root / "active_candidate.json",
        {**body, "checkpoint_sha256": canonical_sha256(body)},
    )


def initialize_research_root(
    root: Path,
    *,
    recorded_at_utc: datetime | str,
    implementation_sha256: str,
) -> tuple[dict[str, Any], bool]:
    payload = {
        "implementation_sha256": _sha(
            implementation_sha256, "implementation_sha256"
        ),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "candidate_ledger.jsonl"
    lock_path = root / ".candidate-lifecycle.lock"
    with lock_path.open("a+", encoding="utf-8") as lifecycle_lock:
        fcntl.flock(lifecycle_lock.fileno(), fcntl.LOCK_EX)
        try:
            row, appended = _append_event(
                ledger_path,
                contract=CANDIDATE_LEDGER_CONTRACT,
                event_type="SYSTEM_BOOTSTRAPPED",
                payload=payload,
                recorded_at_utc=recorded_at_utc,
                identity=canonical_sha256(payload),
            )
            with ledger_path.open("r", encoding="utf-8") as handle:
                rows = _read_ledger(handle, CANDIDATE_LEDGER_CONTRACT)
            _write_active_checkpoint(
                root,
                _candidate_state(rows),
                rows[-1]["event_sha256"],
            )
            return row, appended
        finally:
            fcntl.flock(lifecycle_lock.fileno(), fcntl.LOCK_UN)


def append_candidate_event(
    root: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
    recorded_at_utc: datetime | str,
) -> tuple[dict[str, Any], bool]:
    """Append one validated transition and update its active checkpoint."""

    if event_type not in CANDIDATE_EVENTS - {"SYSTEM_BOOTSTRAPPED"}:
        raise DojoAutonomousEvidenceError("candidate event type is invalid")
    ledger_path = root / "candidate_ledger.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".candidate-lifecycle.lock"
    with lock_path.open("a+", encoding="utf-8") as lifecycle_lock:
        fcntl.flock(lifecycle_lock.fileno(), fcntl.LOCK_EX)
        try:
            if ledger_path.exists():
                with ledger_path.open("r", encoding="utf-8") as handle:
                    rows = _read_ledger(handle, CANDIDATE_LEDGER_CONTRACT)
            else:
                rows = []
            active = _candidate_state(rows)
            event_payload = dict(payload)
            _paper_guard(event_payload, "candidate event")
            if event_type == "CANDIDATE_PREREGISTERED":
                spec = build_candidate_spec(event_payload.get("spec") or {})
                previous_registration = next(
                    (
                        row
                        for row in rows
                        if row.get("event_type") == "CANDIDATE_PREREGISTERED"
                        and row.get("payload", {}).get("candidate_id")
                        == spec["candidate_id"]
                    ),
                    None,
                )
                if previous_registration is not None:
                    return previous_registration, False
                if active is not None:
                    raise DojoAutonomousEvidenceError(
                        "an active candidate already exists"
                    )
                event_payload["spec"] = spec
                event_payload["candidate_id"] = spec["candidate_id"]
                candidate_dir = root / "candidates" / spec["candidate_id"]
                candidate_dir.mkdir(parents=True, exist_ok=False)
                _atomic_json(candidate_dir / "spec.json", spec)
            else:
                candidate_id = _required_text(
                    event_payload.get("candidate_id"), "candidate_id"
                )
                if active is None or candidate_id != active["candidate_id"]:
                    raise DojoAutonomousEvidenceError(
                        "candidate event does not bind the active candidate"
                    )
                if event_type == "REPLAY_STARTED":
                    job = event_payload.get("job_lock")
                    if not isinstance(job, Mapping):
                        raise DojoAutonomousEvidenceError(
                            "REPLAY_STARTED requires a job_lock"
                        )
                    for field in (
                        "git_head_sha256",
                        "spec_sha256",
                        "policy_sha256",
                        "output_manifest_sha256",
                    ):
                        _sha(job.get(field), field)
                    argv = job.get("argv")
                    if not isinstance(argv, list) or not argv:
                        raise DojoAutonomousEvidenceError(
                            "job argv must not be empty"
                        )
                    allowlist = job.get("environment_allowlist")
                    if not isinstance(allowlist, list):
                        raise DojoAutonomousEvidenceError(
                            "job environment allowlist must be a list"
                        )
                    _required_text(job.get("output_directory"), "output_directory")
                    screen = _required_text(job.get("screen_name"), "screen_name")
                    if not screen.startswith("qr-dojo-improve-"):
                        raise DojoAutonomousEvidenceError(
                            "job screen must use the bounded improvement prefix"
                        )
                    pid = job.get("pid")
                    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                        raise DojoAutonomousEvidenceError("job pid must be positive")
                    _sha(job.get("process_command_sha256"), "process_command_sha256")
                elif event_type == "REPLAY_REJECTED":
                    if event_payload.get("death_code") not in DEATH_CODES:
                        raise DojoAutonomousEvidenceError(
                            "rejection death code is invalid"
                        )
                    _required_text(
                        event_payload.get("reason"), "rejection reason"
                    )
                elif event_type == "REPLAY_PASSED":
                    metrics = event_payload.get("independent_stress_metrics")
                    if not isinstance(metrics, Mapping):
                        raise DojoAutonomousEvidenceError("pass metrics are required")
                    if (
                        _finite(metrics.get("pf"), "pf") < 1.25
                        or _finite(metrics.get("net"), "net") <= 0.0
                        or _finite(metrics.get("expectancy"), "expectancy") <= 0.0
                        or metrics.get("worst_day_not_worse") is not True
                        or metrics.get("drawdown_not_worse") is not True
                        or metrics.get("margin_ruin_not_worse") is not True
                        or metrics.get("unresolved_end_exposure") is not False
                    ):
                        raise DojoAutonomousEvidenceError(
                            "independent acceptance gates did not pass"
                        )
                elif event_type == "PAPER_ELIGIBLE":
                    _sha(
                        event_payload.get("implementation_commit_sha256"),
                        "implementation_commit_sha256",
                    )
                    _required_text(
                        event_payload.get("future_experiment_id"),
                        "future_experiment_id",
                    )

            identity = canonical_sha256(
                {"event_type": event_type, "payload": event_payload}
            )
            row, appended = _append_event(
                ledger_path,
                contract=CANDIDATE_LEDGER_CONTRACT,
                event_type=event_type,
                payload=event_payload,
                recorded_at_utc=recorded_at_utc,
                identity=identity,
            )
            with ledger_path.open("r", encoding="utf-8") as handle:
                final_rows = _read_ledger(handle, CANDIDATE_LEDGER_CONTRACT)
            final_active = _candidate_state(final_rows)
            _write_active_checkpoint(
                root, final_active, final_rows[-1]["event_sha256"]
            )
            return row, appended
        finally:
            fcntl.flock(lifecycle_lock.fileno(), fcntl.LOCK_UN)


def validate_research_root(root: Path) -> dict[str, Any]:
    """Validate both ledgers and the active checkpoint without mutation."""

    candidate_path = root / "candidate_ledger.jsonl"
    candidate = validate_event_ledger(
        candidate_path, contract=CANDIDATE_LEDGER_CONTRACT
    )
    shadow = validate_shadow_ledger(root / "ai_shadow_ledger.jsonl")
    if candidate["status"] == "MISSING":
        return {"status": "MISSING", "candidate": candidate, "shadow": shadow}
    with candidate_path.open("r", encoding="utf-8") as handle:
        rows = _read_ledger(handle, CANDIDATE_LEDGER_CONTRACT)
    for row in rows:
        if row.get("event_type") not in CANDIDATE_EVENTS:
            raise DojoAutonomousEvidenceError(
                "candidate ledger event is invalid"
            )
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            raise DojoAutonomousEvidenceError("candidate payload is invalid")
        _paper_guard(payload, "candidate payload")
        event_type = row["event_type"]
        expected_identity = canonical_sha256(
            {"event_type": event_type, "payload": payload}
        )
        if event_type == "SYSTEM_BOOTSTRAPPED":
            expected_identity = canonical_sha256(payload)
            _sha(payload.get("implementation_sha256"), "implementation_sha256")
        if row.get("identity") != expected_identity:
            raise DojoAutonomousEvidenceError(
                "candidate ledger identity is invalid"
            )
        if event_type == "CANDIDATE_PREREGISTERED":
            spec = build_candidate_spec(payload.get("spec") or {})
            if (
                spec != payload.get("spec")
                or payload.get("candidate_id") != spec["candidate_id"]
            ):
                raise DojoAutonomousEvidenceError(
                    "candidate registration is not canonical"
                )
            spec_path = root / "candidates" / spec["candidate_id"] / "spec.json"
            if not spec_path.is_file() or json.loads(
                spec_path.read_text(encoding="utf-8")
            ) != spec:
                raise DojoAutonomousEvidenceError(
                    "candidate immutable spec is missing or changed"
                )
        elif event_type == "REPLAY_REJECTED":
            if payload.get("death_code") not in DEATH_CODES:
                raise DojoAutonomousEvidenceError(
                    "candidate rejection death code is invalid"
                )
            _required_text(payload.get("reason"), "candidate rejection reason")
        elif event_type == "REPLAY_STARTED":
            job = payload.get("job_lock")
            if not isinstance(job, Mapping):
                raise DojoAutonomousEvidenceError("candidate job lock is invalid")
            for field in (
                "git_head_sha256",
                "spec_sha256",
                "policy_sha256",
                "output_manifest_sha256",
                "process_command_sha256",
            ):
                _sha(job.get(field), field)
            if not isinstance(job.get("argv"), list) or not job.get("argv"):
                raise DojoAutonomousEvidenceError("candidate job argv is invalid")
            if not isinstance(job.get("environment_allowlist"), list):
                raise DojoAutonomousEvidenceError(
                    "candidate environment allowlist is invalid"
                )
            _required_text(job.get("output_directory"), "output_directory")
            if not _required_text(
                job.get("screen_name"), "screen_name"
            ).startswith("qr-dojo-improve-"):
                raise DojoAutonomousEvidenceError("candidate screen is invalid")
            pid = job.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise DojoAutonomousEvidenceError("candidate pid is invalid")
        elif event_type == "REPLAY_PASSED":
            metrics = payload.get("independent_stress_metrics")
            if not isinstance(metrics, Mapping) or (
                _finite(metrics.get("pf"), "pf") < 1.25
                or _finite(metrics.get("net"), "net") <= 0.0
                or _finite(metrics.get("expectancy"), "expectancy") <= 0.0
                or metrics.get("worst_day_not_worse") is not True
                or metrics.get("drawdown_not_worse") is not True
                or metrics.get("margin_ruin_not_worse") is not True
                or metrics.get("unresolved_end_exposure") is not False
            ):
                raise DojoAutonomousEvidenceError(
                    "candidate stored pass does not meet acceptance gates"
                )
        elif event_type == "PAPER_ELIGIBLE":
            _sha(
                payload.get("implementation_commit_sha256"),
                "implementation_commit_sha256",
            )
            _required_text(
                payload.get("future_experiment_id"), "future_experiment_id"
            )
    active = _candidate_state(rows)
    checkpoint_path = root / "active_candidate.json"
    if not checkpoint_path.exists():
        raise DojoAutonomousEvidenceError(
            "active candidate checkpoint is missing"
        )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    body = {
        key: value
        for key, value in checkpoint.items()
        if key != "checkpoint_sha256"
    }
    if (
        checkpoint.get("checkpoint_sha256") != canonical_sha256(body)
        or checkpoint.get("contract") != ACTIVE_CHECKPOINT_CONTRACT
        or checkpoint.get("candidate_ledger_tip_sha256")
        != candidate["tip_sha256"]
        or checkpoint.get("active_candidate") != active
    ):
        raise DojoAutonomousEvidenceError(
            "active candidate checkpoint is invalid"
        )
    _paper_guard(checkpoint, "active checkpoint")
    return {
        "status": "VALID",
        "candidate": candidate,
        "shadow": shadow,
        "active_candidate": active,
    }
