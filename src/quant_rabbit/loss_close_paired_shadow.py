"""Read-only primitives for a loss-side CLOSE paired shadow cohort.

This module deliberately has no ledger, filesystem, network, or broker
dependencies.  It only validates one frozen decision-time state and compares a
GPT close with the first frozen TP/SL touch in complete S5 bid/ask candles.

Digest fields and supplied S5 candles are not authoritative artifacts by
themselves.  This pure module cannot verify their origin, so a successful
calculation is explicitly unverified, proof-ineligible, and never live
permission.  A future adapter must verify every referenced artifact and use an
immutable event UID for identity; ``state_sha256`` is only a content hash.

Bid/ask spread is represented by the executable prices themselves: LONG exits
are evaluated on bid and SHORT exits on ask.  The explicit cost inputs are
therefore *non-spread* costs and must not contain another spread charge.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


PAIRED_SHADOW_STATE_CONTRACT = "paired_shadow_state_v1"
PAIRED_SHADOW_STATE_VALIDATION_CONTRACT = "paired_shadow_state_validation_v1"
PAIRED_SHADOW_SCORE_CONTRACT = "loss_close_paired_shadow_score_v1"
S5_GRANULARITY_SECONDS = 5
# Match RiskPolicy.max_quote_age_seconds.  Keep the comparison in integer
# nanoseconds so the 20-second boundary cannot be softened by float rounding.
MAX_PAIRED_SHADOW_QUOTE_AGE_SECONDS = 20
_MAX_PAIRED_SHADOW_QUOTE_AGE_NANOSECONDS = (
    MAX_PAIRED_SHADOW_QUOTE_AGE_SECONDS * 1_000_000_000
)
MAX_PAIRED_SHADOW_FILL_DELAY_SECONDS = 20
_MAX_PAIRED_SHADOW_FILL_DELAY_NANOSECONDS = (
    MAX_PAIRED_SHADOW_FILL_DELAY_SECONDS * 1_000_000_000
)
# Broker units are an integer lifecycle identity, not an arbitrary-precision
# math input.  A signed 64-bit ceiling prevents int-to-float overflow before
# the downstream finite-result checks run.
MAX_PAIRED_SHADOW_UNITS = (1 << 63) - 1

_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRADE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_EVENT_UID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_RFC3339_UTC_RE = re.compile(
    r"^(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)
_STATE_KEYS = frozenset(
    {
        "contract",
        "trade_id",
        "close_decision_event_uid",
        "pair",
        "side",
        "units",
        "decision_timestamp_utc",
        "quote_timestamp_utc",
        "decision_bid",
        "decision_ask",
        "executable_close_price",
        "take_profit",
        "stop_loss",
        "quote_to_jpy",
        "broker_snapshot_sha256",
        "decision_unrealized_pnl_jpy",
        "close_verifier_receipt_sha256",
        "close_verifier_verdict",
        "technical_context_sha256",
        "cost_surface_sha256",
        "take_profit_exit_non_spread_cost_jpy",
        "stop_loss_exit_non_spread_cost_jpy",
        "control_financing_stress_jpy",
        "read_only",
        "live_permission_allowed",
        "state_sha256",
    }
)
_NUMERIC_STATE_KEYS = (
    "decision_bid",
    "decision_ask",
    "executable_close_price",
    "take_profit",
    "stop_loss",
    "quote_to_jpy",
    "decision_unrealized_pnl_jpy",
    "take_profit_exit_non_spread_cost_jpy",
    "stop_loss_exit_non_spread_cost_jpy",
    "control_financing_stress_jpy",
)


@dataclass(frozen=True)
class S5Ohlc:
    """One side of a complete S5 bid/ask candle."""

    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class S5BidAskCandle:
    """A broker S5 candle carrying separate bid and ask OHLC truth."""

    timestamp_utc: datetime
    pair: str
    bid: S5Ohlc
    ask: S5Ohlc
    complete: bool = True


def _snapshot_state_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Take one detached plain-dict read of an untrusted mapping."""

    try:
        return dict(value)
    except Exception as exc:
        raise ValueError("paired shadow state snapshot is unreadable") from exc


def _state_keys_are_exact_strings(value: Mapping[object, object]) -> bool:
    return all(_is_exact_str(key) for key in value)


def _is_exact_str(value: object) -> bool:
    return value.__class__ is str


def _is_exact_float(value: object) -> bool:
    return value.__class__ is float


def _is_exact_int(value: object) -> bool:
    return value.__class__ is int


def paired_shadow_state_sha256(value: Mapping[str, Any]) -> str:
    """Return the strict canonical digest, excluding ``state_sha256`` itself."""

    if not isinstance(value, Mapping):
        raise TypeError("paired shadow state must be a mapping")
    snapshot = _snapshot_state_mapping(value)
    if not _state_keys_are_exact_strings(snapshot):
        raise ValueError("paired shadow state keys must be exact strings")
    body = {key: item for key, item in snapshot.items() if key != "state_sha256"}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_paired_shadow_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy, hash, and validate a complete ``paired_shadow_state_v1`` body.

    This is a pure construction helper for decision-time callers.  It neither
    persists the state nor grants live permission.
    """

    if not isinstance(value, Mapping):
        raise TypeError("paired shadow state must be a mapping")
    sealed = _snapshot_state_mapping(value)
    sealed.pop("state_sha256", None)
    for key in _NUMERIC_STATE_KEYS:
        parsed = _finite_number(sealed.get(key))
        if parsed is not None:
            sealed[key] = 0.0 if parsed == 0.0 else parsed
    for key in ("decision_timestamp_utc", "quote_timestamp_utc"):
        parts = _parse_canonical_utc_parts(sealed.get(key))
        if parts is not None:
            sealed[key] = _canonical_utc_from_parts(parts)
    sealed["state_sha256"] = paired_shadow_state_sha256(sealed)
    validation = validate_paired_shadow_state(sealed)
    if not validation["valid"]:
        raise ValueError(
            "invalid paired shadow state: " + ", ".join(validation["issues"])
        )
    return sealed


def validate_paired_shadow_state(value: object) -> dict[str, Any]:
    """Validate a frozen state without trusting its stored digest.

    The function returns a read-only diagnostic instead of raising on malformed
    external input.  Unknown fields are rejected so a caller cannot quietly
    change the economic contract while retaining the v1 label.
    """

    issues: list[str] = []
    computed_sha: str | None = None
    stored_sha: str | None = None
    quote_age_nanoseconds: int | None = None
    if not isinstance(value, Mapping):
        issues.append("STATE_NOT_MAPPING")
        return _validation_result(
            issues,
            stored_sha=None,
            computed_sha=None,
            quote_age_nanoseconds=None,
        )

    try:
        value = _snapshot_state_mapping(value)
    except ValueError:
        issues.append("STATE_SNAPSHOT_UNREADABLE")
        return _validation_result(
            issues,
            stored_sha=None,
            computed_sha=None,
            quote_age_nanoseconds=None,
        )

    if not _state_keys_are_exact_strings(value):
        issues.append("NON_STRING_STATE_KEY")
    keys = {key for key in value if _is_exact_str(key)}
    for key in sorted(_STATE_KEYS - keys):
        issues.append(f"MISSING_FIELD:{key}")
    for key in sorted(keys - _STATE_KEYS):
        issues.append(f"UNKNOWN_FIELD:{key}")

    contract = value.get("contract")
    if not _is_exact_str(contract) or contract != PAIRED_SHADOW_STATE_CONTRACT:
        issues.append("INVALID_CONTRACT")
    if value.get("read_only") is not True:
        issues.append("READ_ONLY_REQUIRED")
    if value.get("live_permission_allowed") is not False:
        issues.append("LIVE_PERMISSION_MUST_BE_FALSE")

    trade_id = value.get("trade_id")
    if not _is_exact_str(trade_id) or _TRADE_ID_RE.fullmatch(trade_id) is None:
        issues.append("INVALID_TRADE_ID")
    decision_event_uid = value.get("close_decision_event_uid")
    if (
        not _is_exact_str(decision_event_uid)
        or _EVENT_UID_RE.fullmatch(decision_event_uid) is None
    ):
        issues.append("INVALID_CLOSE_DECISION_EVENT_UID")
    pair = value.get("pair")
    if not _is_exact_str(pair) or _PAIR_RE.fullmatch(pair) is None:
        issues.append("INVALID_PAIR")
    side = value.get("side")
    if not _is_exact_str(side) or side not in {"LONG", "SHORT"}:
        issues.append("INVALID_SIDE")
    units = value.get("units")
    if not _is_exact_int(units) or not 1 <= units <= MAX_PAIRED_SHADOW_UNITS:
        issues.append("INVALID_UNITS")
    verifier_verdict = value.get("close_verifier_verdict")
    if not _is_exact_str(verifier_verdict) or verifier_verdict != "PASS":
        issues.append("CLOSE_VERIFIER_VERDICT_NOT_PASS")

    timestamps_ns: dict[str, int] = {}
    for key in ("decision_timestamp_utc", "quote_timestamp_utc"):
        parsed = _parse_canonical_utc_parts(value.get(key))
        if parsed is None:
            issues.append(f"INVALID_TIMESTAMP:{key}")
        else:
            timestamps_ns[key] = parsed[1]
            if value.get(key) != _canonical_utc_from_parts(parsed):
                issues.append(f"NON_CANONICAL_TIMESTAMP:{key}")
    if (
        "decision_timestamp_utc" in timestamps_ns
        and "quote_timestamp_utc" in timestamps_ns
    ):
        quote_age_nanoseconds = (
            timestamps_ns["decision_timestamp_utc"]
            - timestamps_ns["quote_timestamp_utc"]
        )
        if quote_age_nanoseconds < 0:
            issues.append("QUOTE_AFTER_DECISION")
        elif quote_age_nanoseconds > _MAX_PAIRED_SHADOW_QUOTE_AGE_NANOSECONDS:
            issues.append("QUOTE_STALE_AT_DECISION")

    numbers: dict[str, float] = {}
    for key in _NUMERIC_STATE_KEYS:
        raw_number = value.get(key)
        number = _finite_number(raw_number)
        if number is None:
            issues.append(f"INVALID_NUMBER:{key}")
        else:
            numbers[key] = number
            if not _is_exact_float(raw_number) or (
                number == 0.0 and math.copysign(1.0, number) < 0.0
            ):
                issues.append(f"NON_CANONICAL_NUMBER:{key}")

    for key in (
        "decision_bid",
        "decision_ask",
        "executable_close_price",
        "take_profit",
        "stop_loss",
        "quote_to_jpy",
    ):
        if key in numbers and numbers[key] <= 0.0:
            issues.append(f"NON_POSITIVE_NUMBER:{key}")
    for key in (
        "take_profit_exit_non_spread_cost_jpy",
        "stop_loss_exit_non_spread_cost_jpy",
        "control_financing_stress_jpy",
    ):
        if key in numbers and numbers[key] < 0.0:
            issues.append(f"NEGATIVE_COST:{key}")
    decision_unrealized = numbers.get("decision_unrealized_pnl_jpy")
    if decision_unrealized is not None and decision_unrealized >= 0.0:
        issues.append("DECISION_NOT_LOSS_SIDE")

    bid = numbers.get("decision_bid")
    ask = numbers.get("decision_ask")
    executable = numbers.get("executable_close_price")
    take_profit = numbers.get("take_profit")
    stop_loss = numbers.get("stop_loss")
    if bid is not None and ask is not None and not bid < ask:
        issues.append("INVALID_BID_ASK_ORDER")
    if (
        executable is not None
        and side == "LONG"
        and bid is not None
        and executable != bid
    ):
        issues.append("LONG_EXECUTABLE_CLOSE_MUST_EQUAL_BID")
    if (
        executable is not None
        and side == "SHORT"
        and ask is not None
        and executable != ask
    ):
        issues.append("SHORT_EXECUTABLE_CLOSE_MUST_EQUAL_ASK")
    if executable is not None and take_profit is not None and stop_loss is not None:
        if side == "LONG" and not stop_loss < executable < take_profit:
            issues.append("INVALID_LONG_PROTECTION_GEOMETRY")
        if side == "SHORT" and not take_profit < executable < stop_loss:
            issues.append("INVALID_SHORT_PROTECTION_GEOMETRY")

    for key in (
        "broker_snapshot_sha256",
        "close_verifier_receipt_sha256",
        "technical_context_sha256",
        "cost_surface_sha256",
    ):
        digest = value.get(key)
        if not _is_exact_str(digest) or _SHA256_RE.fullmatch(digest) is None:
            issues.append(f"INVALID_SHA256:{key}")
    raw_stored_sha = value.get("state_sha256")
    if _is_exact_str(raw_stored_sha):
        stored_sha = raw_stored_sha
    if stored_sha is None or _SHA256_RE.fullmatch(stored_sha) is None:
        issues.append("INVALID_STATE_SHA256")
    try:
        computed_sha = paired_shadow_state_sha256(value)
    except (TypeError, ValueError):
        issues.append("STATE_NOT_CANONICAL_JSON")
    if computed_sha is not None and stored_sha != computed_sha:
        issues.append("STATE_SHA256_MISMATCH")

    return _validation_result(
        issues,
        stored_sha=stored_sha,
        computed_sha=computed_sha,
        quote_age_nanoseconds=quote_age_nanoseconds,
    )


def score_loss_close_paired_shadow(
    state: object,
    candles: Sequence[S5BidAskCandle],
    *,
    gpt_exit_price: float | None = None,
    gpt_exit_non_spread_cost_jpy: float | None = None,
    gpt_financing_cost_jpy: float | None = None,
    gpt_execution_evidence_sha256: str | None = None,
    gpt_fill_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Score one GPT close against frozen TP/SL from the same decision quote.

    Caller-supplied GPT execution price, non-spread cost, financing cost,
    evidence digest, and fill timestamp are all mandatory.  A decision quote
    is never substituted for a missing fill.  The fill must be at or after the
    decision, no more than 20 seconds later, and inside the executable-side
    range of its supplied complete S5 candle.  Control financing stress is
    applied exactly once.  These checks do not authenticate either artifact.
    """

    frozen_state: dict[str, Any] | None = None
    if isinstance(state, Mapping):
        try:
            frozen_state = _snapshot_state_mapping(state)
        except ValueError:
            return _score_result(
                status="BLOCKED",
                state_sha256=None,
                blockers=["STATE_SNAPSHOT_UNREADABLE"],
            )
    validation = validate_paired_shadow_state(
        frozen_state if frozen_state is not None else state
    )
    if not validation["valid"]:
        return _score_result(
            status="BLOCKED",
            state_sha256=validation.get("state_sha256"),
            blockers=[
                f"INVALID_PAIRED_SHADOW_STATE:{issue}" for issue in validation["issues"]
            ],
        )
    assert frozen_state is not None
    state = frozen_state

    gpt_cost = _finite_nonnegative(gpt_exit_non_spread_cost_jpy)
    gpt_financing = _finite_nonnegative(gpt_financing_cost_jpy)
    parsed_gpt_price = _finite_number(gpt_exit_price)
    execution_sha_valid = (
        _is_exact_str(gpt_execution_evidence_sha256)
        and _SHA256_RE.fullmatch(gpt_execution_evidence_sha256) is not None
    )
    fill_time_parts = _parse_canonical_utc_parts(gpt_fill_timestamp_utc)
    blockers: list[str] = []
    if parsed_gpt_price is None or parsed_gpt_price <= 0.0:
        blockers.append("MISSING_OR_INVALID_GPT_EXIT_PRICE")
    if gpt_exit_non_spread_cost_jpy is None or gpt_cost is None:
        blockers.append("MISSING_OR_INVALID_GPT_NON_SPREAD_COST")
    if gpt_financing_cost_jpy is None or gpt_financing is None:
        blockers.append("MISSING_OR_INVALID_GPT_FINANCING_COST")
    if not execution_sha_valid:
        blockers.append("MISSING_OR_INVALID_GPT_EXECUTION_EVIDENCE_SHA256")
    if fill_time_parts is None:
        blockers.append("MISSING_OR_INVALID_GPT_FILL_TIMESTAMP")
    elif gpt_fill_timestamp_utc != _canonical_utc_from_parts(fill_time_parts):
        blockers.append("NON_CANONICAL_GPT_FILL_TIMESTAMP")
    fill_delay_nanoseconds: int | None = None
    decision_parts = _parse_canonical_utc_parts(state["decision_timestamp_utc"])
    if fill_time_parts is not None and decision_parts is not None:
        fill_delay_nanoseconds = fill_time_parts[1] - decision_parts[1]
        if fill_delay_nanoseconds < 0:
            blockers.append("GPT_FILL_BEFORE_DECISION")
        elif fill_delay_nanoseconds > _MAX_PAIRED_SHADOW_FILL_DELAY_NANOSECONDS:
            blockers.append("GPT_FILL_AFTER_MAX_DELAY")
    if blockers:
        return _score_result(
            status="BLOCKED",
            state_sha256=str(state["state_sha256"]),
            blockers=blockers,
        )
    assert parsed_gpt_price is not None
    assert gpt_cost is not None
    assert gpt_financing is not None
    assert gpt_execution_evidence_sha256 is not None
    assert gpt_fill_timestamp_utc is not None
    assert fill_delay_nanoseconds is not None
    gpt_price = parsed_gpt_price

    frozen_candles, control_candles, candle_blockers = _validated_s5_candles(
        candles,
        state=state,
        decision_timestamp_parts=_parse_canonical_utc_parts(
            state["decision_timestamp_utc"]
        ),
    )
    if candle_blockers:
        return _score_result(
            status="BLOCKED",
            state_sha256=str(state["state_sha256"]),
            blockers=candle_blockers,
        )

    fill_s5_evidence, fill_s5_blockers = _gpt_fill_s5_evidence(
        state,
        frozen_candles,
        fill_timestamp_parts=fill_time_parts,
        gpt_exit_price=gpt_price,
    )
    if fill_s5_blockers:
        return _score_result(
            status="BLOCKED",
            state_sha256=str(state["state_sha256"]),
            blockers=fill_s5_blockers,
        )
    assert fill_s5_evidence is not None

    first_touch = _first_protection_touch(state, control_candles)
    if first_touch is None:
        return _score_result(
            status="PENDING_CONTROL_RESOLUTION_UNVERIFIED_ARTIFACT_BINDINGS",
            state_sha256=str(state["state_sha256"]),
            blockers=[],
            first_touch=None,
        )
    if first_touch["reason"] == "AMBIGUOUS":
        return _score_result(
            status="BLOCKED",
            state_sha256=str(state["state_sha256"]),
            blockers=["S5_CONTROL_SAME_CANDLE_TOUCH_AMBIGUOUS"],
            first_touch=first_touch,
        )

    side = str(state["side"])
    units = int(state["units"])
    quote_to_jpy = float(state["quote_to_jpy"])
    decision_price = float(state["executable_close_price"])
    control_exit_price = float(first_touch["exit_price"])
    control_exit_cost_key = (
        "take_profit_exit_non_spread_cost_jpy"
        if first_touch["reason"] == "TP"
        else "stop_loss_exit_non_spread_cost_jpy"
    )
    control_exit_cost = float(state[control_exit_cost_key])
    control_financing = float(state["control_financing_stress_jpy"])
    stop_loss = float(state["stop_loss"])
    try:
        gpt_gross = _directional_jpy(
            side,
            start=decision_price,
            end=gpt_price,
            units=units,
            quote_to_jpy=quote_to_jpy,
        )
        control_gross = _directional_jpy(
            side,
            start=decision_price,
            end=control_exit_price,
            units=units,
            quote_to_jpy=quote_to_jpy,
        )
        gpt_net = gpt_gross - gpt_cost - gpt_financing
        control_net = control_gross - control_exit_cost - control_financing
        gross_risk = abs(decision_price - stop_loss) * units * quote_to_jpy
        decision_risk = (
            gross_risk
            + float(state["stop_loss_exit_non_spread_cost_jpy"])
            + control_financing
        )
        delta_jpy = gpt_net - control_net
    except (OverflowError, ValueError):
        return _score_result(
            status="BLOCKED",
            state_sha256=str(state["state_sha256"]),
            blockers=["ECONOMIC_CALCULATION_OVERFLOW"],
            first_touch=first_touch,
        )
    computed = (
        gpt_gross,
        control_gross,
        gpt_net,
        control_net,
        gross_risk,
        decision_risk,
        delta_jpy,
    )
    if not all(math.isfinite(value) for value in computed) or decision_risk <= 0.0:
        return _score_result(
            status="BLOCKED",
            state_sha256=str(state["state_sha256"]),
            blockers=["NON_FINITE_OR_NON_POSITIVE_DECISION_RISK"],
            first_touch=first_touch,
        )
    delta_r = delta_jpy / decision_risk
    if not math.isfinite(delta_r):
        return _score_result(
            status="BLOCKED",
            state_sha256=str(state["state_sha256"]),
            blockers=["NON_FINITE_DELTA_R"],
            first_touch=first_touch,
        )

    return _score_result(
        status="CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS",
        state_sha256=str(state["state_sha256"]),
        blockers=[],
        first_touch=first_touch,
        gpt_arm={
            "exit_price": gpt_price,
            "exit_price_source": "CALLER_SUPPLIED_UNVERIFIED_EXECUTION_FIELDS",
            "execution_evidence_sha256": gpt_execution_evidence_sha256,
            "fill_timestamp_utc": gpt_fill_timestamp_utc,
            "fill_delay_nanoseconds": fill_delay_nanoseconds,
            "fill_delay_seconds": fill_delay_nanoseconds / 1_000_000_000,
            "max_fill_delay_seconds": MAX_PAIRED_SHADOW_FILL_DELAY_SECONDS,
            "s5_executable_range_evidence": fill_s5_evidence,
            "gross_incremental_jpy": gpt_gross,
            "exit_non_spread_cost_jpy": gpt_cost,
            "financing_cost_jpy": gpt_financing,
            "net_incremental_jpy": gpt_net,
        },
        frozen_control_arm={
            "exit_price": control_exit_price,
            "exit_reason": first_touch["reason"],
            "gross_incremental_jpy": control_gross,
            "exit_non_spread_cost_jpy": control_exit_cost,
            "financing_stress_jpy": control_financing,
            "financing_application_count": 1,
            "net_incremental_jpy": control_net,
        },
        decision_risk={
            "gross_stop_distance_jpy": gross_risk,
            "stop_loss_exit_non_spread_cost_jpy": float(
                state["stop_loss_exit_non_spread_cost_jpy"]
            ),
            "financing_stress_jpy": control_financing,
            "total_jpy": decision_risk,
        },
        delta_jpy=delta_jpy,
        delta_R=delta_r,
    )


def _validation_result(
    issues: Sequence[str],
    *,
    stored_sha: str | None,
    computed_sha: str | None,
    quote_age_nanoseconds: int | None,
) -> dict[str, Any]:
    return {
        "contract": PAIRED_SHADOW_STATE_VALIDATION_CONTRACT,
        "valid": not issues,
        "issues": list(dict.fromkeys(issues)),
        "state_sha256": stored_sha,
        "computed_state_sha256": computed_sha,
        "quote_age_nanoseconds": quote_age_nanoseconds,
        "quote_age_seconds": (
            quote_age_nanoseconds / 1_000_000_000
            if quote_age_nanoseconds is not None
            else None
        ),
        "max_quote_age_seconds": MAX_PAIRED_SHADOW_QUOTE_AGE_SECONDS,
        "schema_validation_only": True,
        "artifact_bindings_verified_by_evaluator": False,
        "external_artifact_verification_required": True,
        "proof_eligible": False,
        "state_sha_is_content_hash_not_event_identity": True,
        "read_only": True,
        "live_permission_allowed": False,
    }


def _score_result(
    *,
    status: str,
    state_sha256: str | None,
    blockers: Sequence[str],
    first_touch: Mapping[str, Any] | None = None,
    gpt_arm: Mapping[str, Any] | None = None,
    frozen_control_arm: Mapping[str, Any] | None = None,
    decision_risk: Mapping[str, Any] | None = None,
    delta_jpy: float | None = None,
    delta_R: float | None = None,
) -> dict[str, Any]:
    return {
        "contract": PAIRED_SHADOW_SCORE_CONTRACT,
        "status": status,
        "state_sha256": state_sha256,
        "blockers": list(blockers),
        "first_touch": dict(first_touch) if first_touch is not None else None,
        "gpt_arm": dict(gpt_arm) if gpt_arm is not None else None,
        "frozen_control_arm": (
            dict(frozen_control_arm) if frozen_control_arm is not None else None
        ),
        "decision_risk": dict(decision_risk) if decision_risk is not None else None,
        "delta_jpy": delta_jpy,
        "delta_R": delta_R,
        "spread_handling": "INTRINSIC_BID_ASK_PRICES_NO_EXTRA_SPREAD_CHARGE",
        "diagnostic_calculation_only": True,
        "proof_eligible": False,
        "artifact_bindings_verified_by_evaluator": False,
        "broker_snapshot_digest_verified_by_evaluator": False,
        "close_verifier_receipt_digest_verified_by_evaluator": False,
        "technical_context_digest_verified_by_evaluator": False,
        "cost_surface_digest_verified_by_evaluator": False,
        "gpt_execution_evidence_digest_verified_by_evaluator": False,
        "s5_truth_verified_by_evaluator": False,
        "external_artifact_verification_required": True,
        "state_sha_is_content_hash_not_event_identity": True,
        "read_only": True,
        "live_permission_allowed": False,
    }


def _validated_s5_candles(
    candles: Sequence[S5BidAskCandle],
    *,
    state: Mapping[str, Any],
    decision_timestamp_parts: tuple[datetime, int] | None,
) -> tuple[
    tuple[S5BidAskCandle, ...],
    tuple[S5BidAskCandle, ...],
    list[str],
]:
    """Freeze and validate one S5 input, then derive the control-time slice.

    Returning both tuples prevents a mutable or adversarial ``Sequence`` from
    presenting one history during validation and another during fill binding.
    """

    if decision_timestamp_parts is None:
        return (), (), ["INVALID_DECISION_TIMESTAMP"]
    pair = str(state["pair"])
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        return (), (), ["S5_CANDLES_NOT_SEQUENCE"]
    try:
        frozen_input = tuple(candles)
    except Exception:
        return (), (), ["S5_CANDLES_UNREADABLE"]
    blockers: list[str] = []
    by_timestamp: dict[datetime, S5BidAskCandle] = {}
    conflicts: set[datetime] = set()
    for index, candle in enumerate(frozen_input):
        if candle.__class__ is not S5BidAskCandle:
            blockers.append(f"INVALID_S5_CANDLE:{index}")
            continue
        candle_issues = _s5_candle_issues(candle, pair=pair)
        blockers.extend(f"{issue}:{index}" for issue in candle_issues)
        if candle_issues:
            continue
        try:
            canonical_candle = _canonical_s5_candle(candle)
        except (AttributeError, OverflowError, TypeError, ValueError):
            blockers.append(f"INVALID_S5_CANDLE_SNAPSHOT:{index}")
            continue
        canonical_issues = _s5_candle_issues(canonical_candle, pair=pair)
        blockers.extend(f"{issue}:{index}" for issue in canonical_issues)
        if canonical_issues:
            continue
        timestamp = canonical_candle.timestamp_utc
        existing = by_timestamp.get(timestamp)
        if existing is None:
            by_timestamp[timestamp] = canonical_candle
        elif existing != canonical_candle:
            conflicts.add(timestamp)
    blockers.extend(
        f"CONFLICTING_S5_CANDLE:{_iso_utc(timestamp)}"
        for timestamp in sorted(conflicts)
    )
    for timestamp in conflicts:
        by_timestamp.pop(timestamp, None)
    ordered = tuple(by_timestamp[timestamp] for timestamp in sorted(by_timestamp))
    if blockers:
        return (), (), list(dict.fromkeys(blockers))
    if not ordered:
        return (), (), ["S5_TRUTH_EMPTY"]
    decision_nanoseconds = decision_timestamp_parts[1]
    granularity_nanoseconds = S5_GRANULARITY_SECONDS * 1_000_000_000
    decision_candle_nanoseconds = (
        decision_nanoseconds // granularity_nanoseconds
    ) * granularity_nanoseconds
    first_candle_nanoseconds = _datetime_epoch_nanoseconds(ordered[0].timestamp_utc)
    if first_candle_nanoseconds < decision_candle_nanoseconds:
        return (), (), ["S5_TRUTH_PRECEDES_DECISION_CANDLE"]
    if first_candle_nanoseconds > decision_candle_nanoseconds:
        return (), (), ["S5_TRUTH_LEADING_GAP"]
    for previous, current in zip(ordered, ordered[1:]):
        if (
            _datetime_epoch_nanoseconds(current.timestamp_utc)
            - _datetime_epoch_nanoseconds(previous.timestamp_utc)
            != granularity_nanoseconds
        ):
            return (
                (),
                (),
                [
                    "S5_TRUTH_INTERNAL_GAP:"
                    f"{_iso_utc(previous.timestamp_utc)}->{_iso_utc(current.timestamp_utc)}"
                ],
            )
    frozen_ordered = ordered
    if decision_nanoseconds != decision_candle_nanoseconds:
        tp_hit, sl_hit, _quote_side = _protection_hits(state, ordered[0])
        if tp_hit or sl_hit:
            return (), (), ["S5_DECISION_CANDLE_TOUCH_AMBIGUOUS"]
        # The decision falls inside this complete S5 candle.  With no barrier
        # touch anywhere in the containing candle, it is safe to advance to
        # the next candle.  A touched containing candle cannot establish
        # whether the touch happened before or after the decision.
        ordered = ordered[1:]
    return frozen_ordered, ordered, []


def _s5_candle_issues(candle: S5BidAskCandle, *, pair: str) -> list[str]:
    issues: list[str] = []
    if candle.complete is not True:
        issues.append("INCOMPLETE_S5_CANDLE")
    if (
        not _is_exact_str(candle.pair)
        or candle.pair != pair
        or _PAIR_RE.fullmatch(candle.pair) is None
    ):
        issues.append("S5_PAIR_MISMATCH")
    timestamp = candle.timestamp_utc
    if timestamp.__class__ is not datetime:
        issues.append("INVALID_S5_TIMESTAMP")
    elif timestamp.tzinfo is None or timestamp.utcoffset() is None:
        issues.append("S5_TIMESTAMP_NOT_UTC_AWARE")
    else:
        timestamp = timestamp.astimezone(timezone.utc)
        if timestamp.microsecond != 0 or timestamp.second % S5_GRANULARITY_SECONDS != 0:
            issues.append("S5_TIMESTAMP_NOT_ALIGNED")
    for label, ohlc in (("BID", candle.bid), ("ASK", candle.ask)):
        if ohlc.__class__ is not S5Ohlc:
            issues.append(f"INVALID_{label}_OHLC")
            continue
        values = (ohlc.open, ohlc.high, ohlc.low, ohlc.close)
        if not all(
            _is_exact_float(value) and math.isfinite(value) and value > 0.0
            for value in values
        ):
            issues.append(f"INVALID_{label}_OHLC")
            continue
        if ohlc.high < max(ohlc.open, ohlc.low, ohlc.close) or ohlc.low > min(
            ohlc.open, ohlc.high, ohlc.close
        ):
            issues.append(f"INVALID_{label}_OHLC_ORDER")
    if not any(issue.startswith("INVALID_BID_OHLC") for issue in issues) and not any(
        issue.startswith("INVALID_ASK_OHLC") for issue in issues
    ):
        if any(
            getattr(candle.ask, field) <= getattr(candle.bid, field)
            for field in ("open", "high", "low", "close")
        ):
            issues.append("INVALID_S5_BID_ASK_SPREAD")
    return issues


def _canonical_s5_candle(candle: S5BidAskCandle) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=candle.timestamp_utc.astimezone(timezone.utc),
        pair=candle.pair,
        bid=S5Ohlc(
            open=float(candle.bid.open),
            high=float(candle.bid.high),
            low=float(candle.bid.low),
            close=float(candle.bid.close),
        ),
        ask=S5Ohlc(
            open=float(candle.ask.open),
            high=float(candle.ask.high),
            low=float(candle.ask.low),
            close=float(candle.ask.close),
        ),
        complete=True,
    )


def _gpt_fill_s5_evidence(
    state: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    fill_timestamp_parts: tuple[datetime, int] | None,
    gpt_exit_price: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Bind a caller-supplied fill to its executable-side S5 price range.

    This proves only internal consistency with the supplied candle sequence.
    It does not establish that either the fill receipt or the candles came from
    an authoritative broker artifact.
    """

    if fill_timestamp_parts is None:
        return None, ["MISSING_OR_INVALID_GPT_FILL_TIMESTAMP"]
    granularity_nanoseconds = S5_GRANULARITY_SECONDS * 1_000_000_000
    fill_nanoseconds = fill_timestamp_parts[1]
    fill_candle_nanoseconds = (
        fill_nanoseconds // granularity_nanoseconds
    ) * granularity_nanoseconds
    matching: list[S5BidAskCandle] = []
    for candle in candles:
        if (
            candle.__class__ is S5BidAskCandle
            and _datetime_epoch_nanoseconds(candle.timestamp_utc)
            == fill_candle_nanoseconds
        ):
            matching.append(candle)
    if not matching:
        return None, ["S5_GPT_FILL_CANDLE_MISSING"]
    candle = matching[0]
    side = str(state["side"])
    quote_side = "BID" if side == "LONG" else "ASK"
    ohlc = candle.bid if quote_side == "BID" else candle.ask
    if not ohlc.low <= gpt_exit_price <= ohlc.high:
        return None, ["GPT_FILL_PRICE_OUTSIDE_S5_EXECUTABLE_RANGE"]
    return (
        {
            "candle_timestamp_utc": _iso_utc(candle.timestamp_utc),
            "quote_side": quote_side,
            "low": float(ohlc.low),
            "high": float(ohlc.high),
            "fill_price_inside_range": True,
            "authoritative_truth_verified_by_evaluator": False,
        },
        [],
    )


def _first_protection_touch(
    state: Mapping[str, Any], candles: Sequence[S5BidAskCandle]
) -> dict[str, Any] | None:
    take_profit = float(state["take_profit"])
    stop_loss = float(state["stop_loss"])
    for candle in candles:
        tp_hit, sl_hit, quote_side = _protection_hits(state, candle)
        # S5 does not prove the order inside one candle.  Picking SL first
        # makes the control arm artificially weak and can reverse delta_R, so
        # a dual touch is never reduced to one scored outcome.
        if tp_hit and sl_hit:
            return {
                "reason": "AMBIGUOUS",
                "candle_timestamp_utc": _iso_utc(candle.timestamp_utc),
                "quote_side": quote_side,
                "exit_price": None,
                "same_candle_tp_and_sl": True,
                "same_candle_policy": "BLOCK_UNORDERED_S5_TOUCH",
            }
        if sl_hit:
            reason = "SL"
            exit_price = stop_loss
        elif tp_hit:
            reason = "TP"
            exit_price = take_profit
        else:
            continue
        return {
            "reason": reason,
            "candle_timestamp_utc": _iso_utc(candle.timestamp_utc),
            "quote_side": quote_side,
            "exit_price": exit_price,
            "same_candle_tp_and_sl": False,
            "same_candle_policy": "NOT_APPLICABLE_SINGLE_TOUCH",
        }
    return None


def _protection_hits(
    state: Mapping[str, Any], candle: S5BidAskCandle
) -> tuple[bool, bool, str]:
    side = str(state["side"])
    take_profit = float(state["take_profit"])
    stop_loss = float(state["stop_loss"])
    if side == "LONG":
        return (
            candle.bid.high >= take_profit,
            candle.bid.low <= stop_loss,
            "BID",
        )
    return (
        candle.ask.low <= take_profit,
        candle.ask.high >= stop_loss,
        "ASK",
    )


def _directional_jpy(
    side: str,
    *,
    start: float,
    end: float,
    units: int,
    quote_to_jpy: float,
) -> float:
    price_delta = end - start if side == "LONG" else start - end
    return price_delta * units * quote_to_jpy


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_nonnegative(value: object) -> float | None:
    parsed = _finite_number(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def _parse_canonical_utc_parts(value: object) -> tuple[datetime, int] | None:
    if not _is_exact_str(value):
        return None
    match = _RFC3339_UTC_RE.fullmatch(value)
    if match is None:
        return None
    try:
        seconds = datetime.strptime(
            match.group("seconds"), "%Y-%m-%dT%H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    fraction = match.group("fraction") or ""
    nanosecond = int(fraction.ljust(9, "0")) if fraction else 0
    parsed = seconds.replace(microsecond=nanosecond // 1_000)
    epoch_delta = seconds - datetime(1970, 1, 1, tzinfo=timezone.utc)
    epoch_seconds = epoch_delta.days * 86_400 + epoch_delta.seconds
    return parsed, epoch_seconds * 1_000_000_000 + nanosecond


def _canonical_utc_from_parts(parts: tuple[datetime, int]) -> str:
    parsed, epoch_nanoseconds = parts
    base = (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%S")
    )
    nanosecond = epoch_nanoseconds % 1_000_000_000
    if not nanosecond:
        return f"{base}Z"
    fraction = f"{nanosecond:09d}".rstrip("0")
    return f"{base}.{fraction}Z"


def _datetime_epoch_nanoseconds(value: datetime) -> int:
    utc = value.astimezone(timezone.utc)
    delta = utc - datetime(1970, 1, 1, tzinfo=timezone.utc)
    whole_seconds = delta.days * 86_400 + delta.seconds
    return whole_seconds * 1_000_000_000 + utc.microsecond * 1_000


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
