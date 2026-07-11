"""Honest, read-only measurement for GPT market-read predictions.

Schema-v1 market-read rows were scored by copying the next verifier run's
current quote into every due horizon.  That made a late observation look like
the exact 30-minute or two-hour close and even labelled endpoint invalidation
as ``INVALIDATED_FIRST`` without a price path.  Schema v2 deliberately keeps
measurement separate from execution:

* schema-v1 lines are retained byte-for-byte as legacy evidence;
* one broker snapshot/pair is one source observation;
* exact repeats coalesce and conflicting reads from the same source snapshot
  are retained but score-ineligible;
* only complete local M5 candle windows can resolve a horizon;
* M5 candles are mid-price diagnostics, never live permission or bid/ask proof;
* direction, target completion, invalidation and full-read completion remain
  separate outcomes.

This module has no broker client and no order/gateway imports.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from quant_rabbit.decision_execution_lineage import (
    DEFAULT_MARKET_READ_EXECUTION_LINKS,
    DecisionExecutionLineageError,
    read_execution_links,
)

from quant_rabbit.guardian_events import DEFAULT_ROUTER_SNAPSHOT_MAX_AGE_SECONDS


# Ledger contract identity.  It is constant so readers can distinguish the
# immutable legacy layout; replace it only with a documented schema migration.
MARKET_READ_SCHEMA_VERSION = 2
MARKET_READ_CONTRACT = "MARKET_READ_LEDGER_V2"
MARKET_READ_TRUTH_SOURCE = "MID_CANDLE_DIAGNOSTIC"
MARKET_READ_REACTION_CONTRACT = "MARKET_READ_REACTION_CHAIN_V1"

# SI time-unit definition, not a strategy threshold.  Replace only if the
# source timestamps stop using standard seconds/minutes.
SECONDS_PER_MINUTE = 60

# Market reality: pair_charts publishes canonical five-minute candles.  The
# value is a format identity, not a tunable trading threshold.  Replace it only
# if the canonical diagnostic truth granularity changes.
M5_SECONDS = 5 * SECONDS_PER_MINUTE

# Product contract: the receipt explicitly predicts the next 30 minutes and
# next two hours.  These are schema fields, not strategy thresholds.
HORIZONS_MINUTES: tuple[tuple[str, int], ...] = (("30m", 30), ("2h", 120))
HORIZON_MINUTES_BY_NAME = dict(HORIZONS_MINUTES)

# Forward scoring uses the established five-minute read-only broker-snapshot
# window.  This is intentionally separate from RiskPolicy's 20-second
# pre-POST execution quote contract: a diagnostic market read may remain
# measurable without ever granting permission to trade a stale quote.
MAX_SOURCE_LAG_SECONDS = DEFAULT_ROUTER_SNAPSHOT_MAX_AGE_SECONDS

# OANDA spot-FX pip convention used only for diagnostic excursion reporting.
# Replace these constants with broker instrument metadata if non-standard pip
# locations or non-FX instruments enter this ledger.
JPY_QUOTED_FX_PIP_SIZE = 0.01
STANDARD_FX_PIP_SIZE = 0.0001

# Diagnostic storage preserves six decimal places, enough for current broker
# FX quotes and JPY accounting without implying execution precision.  Replace
# it with per-instrument/per-currency display precision when that is available.
DIAGNOSTIC_DECIMAL_PLACES = 6

# Human daily reports use one decimal percentage precision.  It is fixed for
# stable review diffs, not a trading threshold; replace it with a report schema
# precision field if consumers need another representation.
REPORT_PERCENT_DECIMAL_PLACES = 1

# Bounded report/packet history prevents unbounded scheduled-task artifacts.
# These are operational context limits, not market filters; replace them with
# explicit packet/report configuration if the context budget changes.
RECENT_REPORT_PREDICTIONS_LIMIT = 12
FEEDBACK_RESOLVED_EXAMPLES_LIMIT = 5

# New local diagnostic artifacts follow the repository's ordinary owner-write,
# world-readable mode. Existing modes are preserved; replace this fallback
# with deployment-owned permission policy if runtime users diverge.
DEFAULT_ARTIFACT_MODE = 0o644
# POSIX permission-bit mask used only to preserve an existing artifact mode;
# replace with platform abstraction if storage stops being POSIX.
FILE_PERMISSION_MASK = 0o777

_LONG_DIRECTIONS = frozenset({"LONG", "BUY", "UP", "BULL", "BULLISH"})
_SHORT_DIRECTIONS = frozenset({"SHORT", "SELL", "DOWN", "BEAR", "BEARISH"})
_RANGE_DIRECTIONS = frozenset({"RANGE"})
_GEOMETRY_ISSUE_CODES = frozenset(
    {
        "MARKET_READ_TARGET_GEOMETRY_CONFLICT",
        "MARKET_READ_INVALIDATION_GEOMETRY_CONFLICT",
        "MARKET_READ_FORCED_TRADE_GEOMETRY_CONFLICT",
    }
)
_FULL_READ_SUCCESS = frozenset({"TARGET_FIRST", "TARGET_ONLY", "RANGE_CONTAINED"})
# A hyphen between two FX prices is a range delimiter, not the sign of the
# second price (``1.1020-1.1030`` must yield two positive values).
_NUMBER_RE = re.compile(r"(?<![\d.])[-+]?\d+(?:\.\d+)?")


class MarketReadLedgerError(RuntimeError):
    """The ledger cannot be safely read or replaced without evidence loss."""


@dataclass
class _DocumentEntry:
    raw: str
    payload: dict[str, Any]
    is_v2: bool


@dataclass(frozen=True)
class _Candle:
    started_at: datetime
    open: float
    high: float
    low: float
    close: float

    @property
    def ended_at(self) -> datetime:
        return self.started_at + timedelta(seconds=M5_SECONDS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _numbers(value: Any) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(str(value or "")):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return values


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _decision_value(decision: Any, name: str, default: Any = None) -> Any:
    if isinstance(decision, Mapping):
        return decision.get(name, default)
    return getattr(decision, name, default)


def _issue_codes(issues: Iterable[Any]) -> list[str]:
    codes: list[str] = []
    for issue in issues:
        code = issue.get("code") if isinstance(issue, Mapping) else getattr(issue, "code", None)
        text = str(code or "").strip()
        if text and text not in codes:
            codes.append(text)
    return codes


def _pair_quote(packet: Mapping[str, Any], pair: str) -> Mapping[str, Any]:
    broker = packet.get("broker_snapshot")
    quotes = broker.get("quotes") if isinstance(broker, Mapping) else None
    quote = quotes.get(pair) if isinstance(quotes, Mapping) else None
    return quote if isinstance(quote, Mapping) else {}


def _current_mid(packet: Mapping[str, Any], pair: str) -> float | None:
    quote = _pair_quote(packet, pair)
    bid = _float(quote.get("bid"))
    ask = _float(quote.get("ask"))
    if bid is not None and ask is not None:
        return round((bid + ask) / 2.0, DIAGNOSTIC_DECIMAL_PLACES)
    if bid is not None:
        return bid
    if ask is not None:
        return ask

    for lane in packet.get("lanes", []) or []:
        if not isinstance(lane, Mapping) or str(lane.get("pair") or "") != pair:
            continue
        technical = lane.get("technical_context") if isinstance(lane.get("technical_context"), Mapping) else {}
        risk = lane.get("risk_metrics") if isinstance(lane.get("risk_metrics"), Mapping) else {}
        for candidate in (technical.get("current_price_mid"), risk.get("entry_price"), lane.get("entry")):
            parsed = _float(candidate)
            if parsed is not None:
                return parsed
    return None


def _source_snapshot_at(packet: Mapping[str, Any]) -> str | None:
    broker = packet.get("broker_snapshot")
    value = broker.get("fetched_at_utc") if isinstance(broker, Mapping) else None
    parsed = _parse_utc(value)
    return _iso(parsed) if parsed is not None else None


def _source_quote_at(packet: Mapping[str, Any], pair: str) -> str | None:
    parsed = _parse_utc(_pair_quote(packet, pair).get("timestamp_utc"))
    return _iso(parsed) if parsed is not None else None


def _snapshot_identity(packet: Mapping[str, Any], pair: str) -> tuple[str | None, dict[str, Any]]:
    broker = packet.get("broker_snapshot")
    broker_payload = dict(broker) if isinstance(broker, Mapping) else {}
    quote = _pair_quote(packet, pair)
    identity = {
        "pair": pair,
        "snapshot_at_utc": _source_snapshot_at(packet),
        "quote_at_utc": _source_quote_at(packet, pair),
        "quote_bid": _float(quote.get("bid")),
        "quote_ask": _float(quote.get("ask")),
        "broker_snapshot_sha256": _digest(broker_payload) if broker_payload else None,
    }
    if not pair or not identity["snapshot_at_utc"]:
        return None, identity
    return _digest(identity), identity


def _prediction_semantics(
    *,
    pair: str,
    direction: str,
    start_price: float | None,
    naked_read: Mapping[str, Any],
    next_30m: Mapping[str, Any],
    next_2h: Mapping[str, Any],
    forced: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pair": pair,
        "direction": direction,
        "start_price": start_price,
        "naked_read": dict(naked_read),
        "next_30m_prediction": dict(next_30m),
        "next_2h_prediction": dict(next_2h),
        "best_trade_if_forced": dict(forced),
    }


def _target_wrong_side(direction: str, basis: float, prices: Sequence[float]) -> bool:
    if not prices:
        return False
    if direction in _LONG_DIRECTIONS:
        return min(prices) <= basis
    if direction in _SHORT_DIRECTIONS:
        return max(prices) >= basis
    return False


def _invalidation_wrong_side(direction: str, basis: float, prices: Sequence[float]) -> bool:
    if not prices:
        return False
    if direction in _LONG_DIRECTIONS:
        return max(prices) >= basis
    if direction in _SHORT_DIRECTIONS:
        return min(prices) <= basis
    return False


def _eligibility_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    issue_codes = {str(code) for code in row.get("verification_issue_codes", []) or [] if str(code)}
    if issue_codes & _GEOMETRY_ISSUE_CODES:
        reasons.append("VERIFIER_GEOMETRY_CONFLICT")
    if not row.get("source_snapshot_identity"):
        reasons.append("SOURCE_SNAPSHOT_IDENTITY_MISSING")
    source_bid = _float(row.get("source_quote_bid"))
    source_ask = _float(row.get("source_quote_ask"))
    if not row.get("source_quote_at_utc"):
        reasons.append("SOURCE_QUOTE_TIMESTAMP_MISSING")
    quote_lag = _float(row.get("source_quote_lag_seconds"))
    snapshot_lag = _float(row.get("source_snapshot_lag_seconds"))
    lag_limit = _float(row.get("source_lag_max_seconds"))
    if lag_limit is None or lag_limit <= 0:
        lag_limit = float(MAX_SOURCE_LAG_SECONDS)
    if quote_lag is not None:
        if quote_lag < 0:
            reasons.append("SOURCE_QUOTE_AFTER_PREDICTION_START")
        elif quote_lag > lag_limit:
            reasons.append("SOURCE_QUOTE_LAG_EXCEEDS_WINDOW")
    if snapshot_lag is not None:
        if snapshot_lag < 0:
            reasons.append("SOURCE_SNAPSHOT_AFTER_PREDICTION_START")
        elif snapshot_lag > lag_limit:
            reasons.append("SOURCE_SNAPSHOT_LAG_EXCEEDS_WINDOW")
    if source_bid is None or source_ask is None or source_bid <= 0 or source_ask <= 0:
        reasons.append("SOURCE_BID_ASK_MISSING")
    elif source_ask < source_bid:
        reasons.append("SOURCE_BID_ASK_INVERTED")
    pair = str(row.get("pair") or "").strip()
    direction = str(row.get("direction") or "").strip().upper()
    start = _float(row.get("start_price"))
    if not pair:
        reasons.append("PAIR_MISSING")
    if direction not in _LONG_DIRECTIONS | _SHORT_DIRECTIONS | _RANGE_DIRECTIONS:
        reasons.append("DIRECTION_UNSCORABLE")
    if start is None or start <= 0:
        reasons.append("START_PRICE_MISSING")
    elif source_bid is not None and source_ask is not None:
        source_mid = round(
            (source_bid + source_ask) / 2.0,
            DIAGNOSTIC_DECIMAL_PLACES,
        )
        if start != source_mid:
            reasons.append("START_PRICE_NOT_SOURCE_BID_ASK_MID")
    if direction in _RANGE_DIRECTIONS:
        for key in ("next_30m_prediction", "next_2h_prediction"):
            prediction = row.get(key) if isinstance(row.get(key), Mapping) else {}
            if len(_numbers(prediction.get("target_zone"))) < 2:
                reasons.append(f"{key.upper()}_RANGE_BOUNDS_MISSING")
    elif start is not None:
        for key in ("next_30m_prediction", "next_2h_prediction"):
            prediction = row.get(key) if isinstance(row.get(key), Mapping) else {}
            predicted_direction = str(prediction.get("direction") or direction).strip().upper()
            targets = _numbers(prediction.get("target_zone"))
            invalidations = _numbers(prediction.get("invalidation"))
            if not targets:
                reasons.append(f"{key.upper()}_TARGET_MISSING")
            if not invalidations:
                reasons.append(f"{key.upper()}_INVALIDATION_MISSING")
            if _target_wrong_side(predicted_direction, start, targets):
                reasons.append(f"{key.upper()}_TARGET_GEOMETRY_CONFLICT")
            if _invalidation_wrong_side(
                predicted_direction,
                start,
                invalidations,
            ):
                reasons.append(f"{key.upper()}_INVALIDATION_GEOMETRY_CONFLICT")
    if row.get("source_snapshot_conflict"):
        reasons.append("SOURCE_SNAPSHOT_PREDICTION_CONFLICT")
    return list(dict.fromkeys(reasons))


def _empty_horizon_result(reason: str = "HORIZON_NOT_DUE") -> dict[str, Any]:
    return {
        "resolution_status": "UNRESOLVED",
        "unresolved_reason": reason,
        "truth_source": MARKET_READ_TRUTH_SOURCE,
        "read_only": True,
        "live_permission": False,
        "direction_status": "UNRESOLVED",
        "target_completion_status": "UNRESOLVED",
        "invalidation_status": "UNRESOLVED",
        "first_touch_status": "UNRESOLVED",
        "full_read_status": "UNRESOLVED",
        "actual_price": None,
        "endpoint_observed_at_utc": None,
        "endpoint_offset_from_due_seconds": None,
        "resolved_at_utc": None,
        "resolution_lag_seconds": None,
        "mfe_pips": None,
        "mae_pips": None,
    }


def _empty_reaction_chain(prediction_id: str) -> dict[str, Any]:
    return {
        "contract": MARKET_READ_REACTION_CONTRACT,
        "prediction_id": prediction_id,
        "prediction_resolution": {
            "status": "UNRESOLVED",
            "score_eligible": None,
            "horizons": {
                "30m": {"status": "UNRESOLVED"},
                "2h": {"status": "UNRESOLVED"},
            },
        },
        "first_subsequent_decision": {
            "status": "UNRESOLVED",
            "decision_receipt_id": None,
            "decision_generated_at_utc": None,
            "decision_recorded_at_utc": None,
            "action": None,
            "verification_status": None,
            "market_read_prediction_id": None,
            "selected_lane_id": None,
            "selected_lane_ids": [],
            "cancel_order_ids": [],
            "close_trade_ids": [],
        },
        "execution_attribution": {
            "status": "UNATTRIBUTED",
            "attribution_basis": None,
            "fill_ids": {"status": "UNATTRIBUTED", "ids": []},
            "order_ids": {"status": "UNATTRIBUTED", "ids": []},
            "trade_ids": {"status": "UNATTRIBUTED", "ids": []},
            "unattributed_reason": "NO_SUBSEQUENT_DECISION_RECEIPT",
        },
        "realized_outcome": {
            "status": "UNRESOLVED",
            "realized_pl_jpy": None,
            "financing_jpy": None,
            "net_realized_jpy": None,
            "trade_outcomes": [],
            "unresolved_reason": "NO_EXACT_TRADE_IDS",
        },
    }


def _empty_direct_execution_attribution(
    *,
    decision_receipt_id: str | None,
    prediction_id: str,
    reason: str = "NO_EXACT_EXECUTION_LINK",
) -> dict[str, Any]:
    return {
        "status": "UNATTRIBUTED",
        "attribution_basis": None,
        "decision_receipt_id": decision_receipt_id,
        "market_read_prediction_id": prediction_id,
        "fill_ids": {"status": "UNATTRIBUTED", "ids": []},
        "order_ids": {"status": "UNATTRIBUTED", "ids": []},
        "trade_ids": {"status": "UNATTRIBUTED", "ids": []},
        "transaction_ids": {"status": "UNATTRIBUTED", "ids": []},
        "pair_or_time_inference_used": False,
        "unattributed_reason": reason,
    }


def _empty_direct_realized_outcome(reason: str = "NO_EXACT_TRADE_IDS") -> dict[str, Any]:
    return {
        "status": "UNRESOLVED",
        "realized_pl_jpy": None,
        "financing_jpy": None,
        "net_realized_jpy": None,
        "trade_outcomes": [],
        "unresolved_reason": reason,
    }


def _build_v2_row(
    decision: Any,
    packet: Mapping[str, Any],
    *,
    status: str,
    issues: Iterable[Any],
    now: datetime,
) -> dict[str, Any] | None:
    market_read = _decision_value(decision, "market_read_first")
    if not isinstance(market_read, Mapping) or not market_read:
        return None
    next_30m = market_read.get("next_30m_prediction") if isinstance(market_read.get("next_30m_prediction"), Mapping) else {}
    next_2h = market_read.get("next_2h_prediction") if isinstance(market_read.get("next_2h_prediction"), Mapping) else {}
    naked = market_read.get("naked_read") if isinstance(market_read.get("naked_read"), Mapping) else {}
    forced = market_read.get("best_trade_if_forced") if isinstance(market_read.get("best_trade_if_forced"), Mapping) else {}
    pair = str(
        next_30m.get("pair")
        or next_2h.get("pair")
        or forced.get("pair")
        or naked.get("cleanest_pair_expression")
        or ""
    ).strip()
    direction = str(
        next_30m.get("direction")
        or next_2h.get("direction")
        or forced.get("direction")
        or ""
    ).strip().upper()
    generated_text = str(_decision_value(decision, "generated_at_utc") or "").strip()
    baseline_generated_at = _parse_utc(generated_text) or now
    provenance = _decision_value(decision, "decision_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    authored_at = _parse_utc(provenance.get("authored_at_utc"))
    applied_at = _parse_utc(provenance.get("applied_at_utc"))
    issue_codes = _issue_codes(issues)
    codex_provenance_valid = (
        status == "ACCEPTED"
        and provenance.get("author_kind") == "CODEX_MARKET_READ"
        and provenance.get("model") == "gpt-5.5"
        and str(provenance.get("reasoning_effort") or "").strip().lower() == "high"
        and applied_at is not None
        and not any(code.startswith("AI_MARKET_READ_") or code.startswith("MARKET_READ_ARTIFACT_") for code in issue_codes)
    )
    # A deterministic baseline exists before Codex has authored the actual
    # forecast.  Starting the diagnostic clock at that earlier timestamp lets
    # the model receive credit for price action it could already see.  The
    # atomic overlay publication time is the first safe forward boundary.
    predicted_at = applied_at if codex_provenance_valid and applied_at is not None else baseline_generated_at
    generated_at = _iso(predicted_at)
    start_price = _current_mid(packet, pair)
    if start_price is None:
        forced_entries = _numbers(forced.get("entry"))
        start_price = forced_entries[0] if forced_entries else None
    source_id, source_fields = _snapshot_identity(packet, pair)
    semantics = _prediction_semantics(
        pair=pair,
        direction=direction,
        start_price=start_price,
        naked_read=naked,
        next_30m=next_30m,
        next_2h=next_2h,
        forced=forced,
    )
    semantic_fingerprint = _digest(semantics)
    prediction_id = "mr2:" + _digest(
        {
            "source_snapshot_identity": (
                source_id
                if source_id is not None
                else {"missing": True, "recorded_at_utc": _iso(now)}
            ),
            "semantic_fingerprint": semantic_fingerprint,
        }
    )
    action = str(_decision_value(decision, "action") or "")
    snapshot_at = _parse_utc(source_fields.get("snapshot_at_utc"))
    quote_at = _parse_utc(source_fields.get("quote_at_utc"))
    snapshot_lag_seconds = (
        (predicted_at - snapshot_at).total_seconds() if snapshot_at is not None else None
    )
    quote_lag_seconds = (
        (predicted_at - quote_at).total_seconds() if quote_at is not None else None
    )
    observation = {
        "generated_at_utc": generated_at,
        "baseline_generated_at_utc": _iso(baseline_generated_at),
        "market_read_authored_at_utc": _iso(authored_at) if authored_at is not None else None,
        "market_read_applied_at_utc": _iso(applied_at) if applied_at is not None else None,
        "prediction_time_basis": "CODEX_APPLIED_AT" if codex_provenance_valid else "DECISION_GENERATED_AT",
        "recorded_at_utc": _iso(now),
        "action": action,
        "verification_status": status,
        "verification_issue_codes": issue_codes,
        "selected_lane_id": _decision_value(decision, "selected_lane_id"),
        "selected_lane_ids": list(_decision_value(decision, "selected_lane_ids", ()) or ()),
    }
    row: dict[str, Any] = {
        "schema_version": MARKET_READ_SCHEMA_VERSION,
        "ledger_contract": MARKET_READ_CONTRACT,
        "prediction_id": prediction_id,
        "semantic_fingerprint": semantic_fingerprint,
        "source_snapshot_identity": source_id,
        "source_snapshot_at_utc": source_fields.get("snapshot_at_utc"),
        "source_quote_at_utc": source_fields.get("quote_at_utc"),
        "source_quote_bid": source_fields.get("quote_bid"),
        "source_quote_ask": source_fields.get("quote_ask"),
        "source_snapshot_lag_seconds": snapshot_lag_seconds,
        "source_quote_lag_seconds": quote_lag_seconds,
        "source_lag_max_seconds": MAX_SOURCE_LAG_SECONDS,
        "source_snapshot_content_sha256": source_fields.get("broker_snapshot_sha256"),
        "generated_at_utc": generated_at,
        "baseline_generated_at_utc": _iso(baseline_generated_at),
        "market_read_authored_at_utc": _iso(authored_at) if authored_at is not None else None,
        "market_read_applied_at_utc": _iso(applied_at) if applied_at is not None else None,
        "prediction_time_basis": "CODEX_APPLIED_AT" if codex_provenance_valid else "DECISION_GENERATED_AT",
        "recorded_at_utc": _iso(now),
        "action": action,
        "verification_status": status,
        "verification_issue_codes": issue_codes,
        "selected_lane_id": _decision_value(decision, "selected_lane_id"),
        "selected_lane_ids": list(_decision_value(decision, "selected_lane_ids", ()) or ()),
        "pair": pair,
        "direction": direction,
        "start_price": start_price,
        "horizon_30m_due_utc": _iso(
            predicted_at + timedelta(minutes=HORIZON_MINUTES_BY_NAME["30m"])
        ),
        "horizon_2h_due_utc": _iso(
            predicted_at + timedelta(minutes=HORIZON_MINUTES_BY_NAME["2h"])
        ),
        "naked_read": dict(naked),
        "next_30m_prediction": dict(next_30m),
        "next_2h_prediction": dict(next_2h),
        "best_trade_if_forced": dict(forced),
        "blocked_but_market_read_recorded": status != "ACCEPTED" or action != "TRADE",
        "truth_source": MARKET_READ_TRUTH_SOURCE,
        "read_only": True,
        "live_permission": False,
        "decision_observations": [observation],
        "duplicate_observation_count": 0,
        "source_snapshot_conflict": False,
        "source_snapshot_conflict_group_id": None,
        "horizon_results": {
            "30m": _empty_horizon_result(),
            "2h": _empty_horizon_result(),
        },
        "actual_30m_price": None,
        "actual_2h_price": None,
        "actual_30m_observed_at_utc": None,
        "actual_2h_observed_at_utc": None,
        "thirty_minute_verdict": "UNRESOLVED",
        "two_hour_verdict": "UNRESOLVED",
        "verdict": "UNRESOLVED",
        "originating_decision_receipt_id": None,
        "direct_execution_attribution": _empty_direct_execution_attribution(
            decision_receipt_id=None,
            prediction_id=prediction_id,
            reason="ORIGINATING_DECISION_RECEIPT_ID_NOT_BOUND",
        ),
        "direct_realized_outcome": _empty_direct_realized_outcome(
            "ORIGINATING_DECISION_RECEIPT_ID_NOT_BOUND"
        ),
        "reaction_chain": _empty_reaction_chain(prediction_id),
    }
    reasons = _eligibility_reasons(row)
    row["score_eligible"] = not reasons
    row["score_ineligible_reasons"] = reasons
    return row


def _read_document(path: Path) -> list[_DocumentEntry]:
    if not path.exists():
        return []
    try:
        # Decode bytes directly so Python's universal-newline reader cannot
        # rewrite legacy CRLF evidence while v2 rows are updated.
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MarketReadLedgerError(f"market-read ledger is not UTF-8: {exc}") from exc
    except OSError as exc:
        raise MarketReadLedgerError(f"market-read ledger unreadable: {exc}") from exc
    entries: list[_DocumentEntry] = []
    for index, raw in enumerate(text.splitlines(keepends=True), start=1):
        if not raw.strip():
            entries.append(_DocumentEntry(raw=raw, payload={}, is_v2=False))
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MarketReadLedgerError(f"market-read ledger malformed at line {index}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MarketReadLedgerError(f"market-read ledger non-object at line {index}")
        entries.append(
            _DocumentEntry(
                raw=raw,
                payload=payload,
                is_v2=payload.get("schema_version") == MARKET_READ_SCHEMA_VERSION,
            )
        )
    return entries


def _serialize_document(entries: Sequence[_DocumentEntry]) -> str:
    parts: list[str] = []
    for entry in entries:
        if entry.is_v2:
            parts.append(json.dumps(entry.payload, ensure_ascii=False, sort_keys=True) + "\n")
            continue
        raw = entry.raw
        if raw and not raw.endswith(("\n", "\r")):
            raise MarketReadLedgerError(
                "legacy market-read row lacks a terminal newline; refusing to alter its bytes"
            )
        parts.append(raw)
    return "".join(parts)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_mode = path.stat().st_mode & FILE_PERMISSION_MASK
    except FileNotFoundError:
        target_mode = DEFAULT_ARTIFACT_MODE
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, target_mode)
        os.replace(temp_path, path)
        temp_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


@contextmanager
def _ledger_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_pair_charts(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _pair_m5_candles(payload: Mapping[str, Any] | None, pair: str) -> tuple[list[_Candle], str | None]:
    if not isinstance(payload, Mapping):
        return [], "PAIR_CHARTS_MISSING_OR_INVALID"
    chart: Mapping[str, Any] | None = None
    for candidate in payload.get("charts", []) or []:
        if isinstance(candidate, Mapping) and str(candidate.get("pair") or "") == pair:
            chart = candidate
            break
    if chart is None:
        return [], "PAIR_M5_CHART_MISSING"
    view: Mapping[str, Any] | None = None
    for candidate in chart.get("views", []) or []:
        if isinstance(candidate, Mapping) and str(candidate.get("granularity") or "").upper() == "M5":
            view = candidate
            break
    if view is None:
        return [], "PAIR_M5_VIEW_MISSING"
    by_time: dict[datetime, _Candle] = {}
    for raw in view.get("recent_candles", []) or []:
        if not isinstance(raw, Mapping) or raw.get("complete") is not True:
            continue
        started = _parse_utc(raw.get("t") or raw.get("timestamp"))
        opened = _float(raw.get("o") or raw.get("open"))
        high = _float(raw.get("h") or raw.get("high"))
        low = _float(raw.get("l") or raw.get("low"))
        close = _float(raw.get("c") or raw.get("close"))
        if started is None or None in {opened, high, low, close}:
            continue
        assert opened is not None and high is not None and low is not None and close is not None
        if high < max(opened, close) or low > min(opened, close) or high < low:
            continue
        candle = _Candle(started, opened, high, low, close)
        previous = by_time.get(started)
        if previous is not None and previous != candle:
            return [], "PAIR_M5_CONFLICTING_DUPLICATE_CANDLE"
        by_time[started] = candle
    if not by_time:
        return [], "PAIR_M5_COMPLETE_CANDLES_MISSING"
    return [by_time[key] for key in sorted(by_time)], None


def _ceil_m5(value: datetime) -> datetime:
    timestamp = value.timestamp()
    remainder = timestamp % M5_SECONDS
    if remainder:
        timestamp += M5_SECONDS - remainder
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _floor_m5(value: datetime) -> datetime:
    timestamp = value.timestamp()
    return datetime.fromtimestamp(timestamp - (timestamp % M5_SECONDS), tz=timezone.utc)


def _complete_window(
    candles: Sequence[_Candle],
    *,
    predicted_at: datetime,
    due_at: datetime,
) -> tuple[list[_Candle], dict[str, Any] | None, str | None]:
    first_start = _ceil_m5(predicted_at)
    last_end = _floor_m5(due_at)
    last_start = last_end - timedelta(seconds=M5_SECONDS)
    if last_start < first_start:
        return [], None, "M5_EFFECTIVE_WINDOW_EMPTY"
    expected: list[datetime] = []
    cursor = first_start
    while cursor <= last_start:
        expected.append(cursor)
        cursor += timedelta(seconds=M5_SECONDS)
    by_time = {candle.started_at: candle for candle in candles}
    missing = [item for item in expected if item not in by_time]
    if missing:
        return [], None, "M5_WINDOW_INCOMPLETE"
    selected = [by_time[item] for item in expected]
    metadata = {
        "window_start_utc": _iso(selected[0].started_at),
        "window_end_utc": _iso(selected[-1].ended_at),
        "candle_count": len(selected),
        "entry_delay_seconds": (selected[0].started_at - predicted_at).total_seconds(),
        "unobserved_horizon_tail_seconds": (due_at - selected[-1].ended_at).total_seconds(),
        "effective_holding_minutes": (
            selected[-1].ended_at - selected[0].started_at
        ).total_seconds()
        / SECONDS_PER_MINUTE,
    }
    return selected, metadata, None


def _touches(
    candles: Sequence[_Candle],
    *,
    direction: str,
    target: float | None,
    invalidation: float | None,
) -> tuple[datetime | None, datetime | None, str]:
    target_at: datetime | None = None
    invalidation_at: datetime | None = None
    for candle in candles:
        if direction in _LONG_DIRECTIONS:
            target_hit = target is not None and candle.high >= target
            invalidation_hit = invalidation is not None and candle.low <= invalidation
        else:
            target_hit = target is not None and candle.low <= target
            invalidation_hit = invalidation is not None and candle.high >= invalidation
        if target_hit and target_at is None:
            target_at = candle.started_at
        if invalidation_hit and invalidation_at is None:
            invalidation_at = candle.started_at
    if target_at is not None and invalidation_at is not None:
        if target_at < invalidation_at:
            order = "TARGET_FIRST"
        elif invalidation_at < target_at:
            order = "INVALIDATION_FIRST"
        else:
            order = "AMBIGUOUS_SAME_CANDLE"
    elif target_at is not None:
        order = "TARGET_ONLY"
    elif invalidation_at is not None:
        order = "INVALIDATION_ONLY"
    else:
        order = "NEITHER_TOUCHED"
    return target_at, invalidation_at, order


def _pip_size(pair: str) -> float:
    return JPY_QUOTED_FX_PIP_SIZE if pair.upper().endswith("_JPY") else STANDARD_FX_PIP_SIZE


def _score_horizon(
    row: Mapping[str, Any],
    *,
    horizon: str,
    minutes: int,
    candles: Sequence[_Candle],
    candle_issue: str | None,
    now: datetime,
) -> dict[str, Any]:
    ineligible_reasons = set(_normalized_score_ineligible_reasons(row))
    if any("GEOMETRY" in reason for reason in ineligible_reasons):
        result = _empty_horizon_result("PREDICTION_GEOMETRY_INVALID")
        result.update(
            {
                "direction_status": "UNSCORABLE_GEOMETRY",
                "target_completion_status": "UNSCORABLE_GEOMETRY",
                "invalidation_status": "UNSCORABLE_GEOMETRY",
                "first_touch_status": "UNSCORABLE_GEOMETRY",
                "full_read_status": "UNSCORABLE_GEOMETRY",
            }
        )
        return result
    predicted_at = _parse_utc(row.get("generated_at_utc"))
    if predicted_at is None:
        return _empty_horizon_result("PREDICTED_AT_INVALID")
    due_at = predicted_at + timedelta(minutes=minutes)
    if now < due_at:
        return _empty_horizon_result("HORIZON_NOT_DUE")
    if candle_issue:
        return _empty_horizon_result(candle_issue)
    window, window_metadata, window_issue = _complete_window(
        candles,
        predicted_at=predicted_at,
        due_at=due_at,
    )
    if window_issue or not window or window_metadata is None:
        return _empty_horizon_result(window_issue or "M5_WINDOW_INCOMPLETE")

    prediction_key = "next_30m_prediction" if horizon == "30m" else "next_2h_prediction"
    prediction = row.get(prediction_key) if isinstance(row.get(prediction_key), Mapping) else {}
    direction = str(prediction.get("direction") or row.get("direction") or "").strip().upper()
    pair = str(row.get("pair") or "")
    start = _float(row.get("start_price"))
    actual = window[-1].close
    endpoint_at = window[-1].ended_at
    result = _empty_horizon_result("")
    result.update(
        {
            "resolution_status": "RESOLVED_MID_CANDLE_DIAGNOSTIC",
            "unresolved_reason": None,
            "actual_price": actual,
            "endpoint_observed_at_utc": _iso(endpoint_at),
            "endpoint_offset_from_due_seconds": (endpoint_at - due_at).total_seconds(),
            "resolved_at_utc": _iso(now),
            "resolution_lag_seconds": max(0.0, (now - due_at).total_seconds()),
            **window_metadata,
        }
    )
    if start is None:
        result.update(
            {
                "direction_status": "UNSCORABLE_START_PRICE",
                "target_completion_status": "UNSCORABLE_START_PRICE",
                "invalidation_status": "UNSCORABLE_START_PRICE",
                "first_touch_status": "UNSCORABLE_START_PRICE",
                "full_read_status": "UNSCORABLE_START_PRICE",
            }
        )
        return result

    pip = _pip_size(pair)
    if direction in _LONG_DIRECTIONS:
        result["direction_status"] = "CORRECT" if actual > start else "WRONG"
        result["mfe_pips"] = round(
            max(0.0, max(candle.high for candle in window) - start) / pip,
            DIAGNOSTIC_DECIMAL_PLACES,
        )
        result["mae_pips"] = round(
            max(0.0, start - min(candle.low for candle in window)) / pip,
            DIAGNOSTIC_DECIMAL_PLACES,
        )
    elif direction in _SHORT_DIRECTIONS:
        result["direction_status"] = "CORRECT" if actual < start else "WRONG"
        result["mfe_pips"] = round(
            max(0.0, start - min(candle.low for candle in window)) / pip,
            DIAGNOSTIC_DECIMAL_PLACES,
        )
        result["mae_pips"] = round(
            max(0.0, max(candle.high for candle in window) - start) / pip,
            DIAGNOSTIC_DECIMAL_PLACES,
        )
    elif direction in _RANGE_DIRECTIONS:
        bounds = _numbers(prediction.get("target_zone"))
        if len(bounds) < 2:
            result.update(
                {
                    "direction_status": "UNSCORABLE_RANGE_BOUNDS",
                    "target_completion_status": "NOT_APPLICABLE_RANGE",
                    "invalidation_status": "NOT_APPLICABLE_RANGE",
                    "first_touch_status": "NOT_APPLICABLE_RANGE",
                    "full_read_status": "UNSCORABLE_RANGE_BOUNDS",
                }
            )
            return result
        lower, upper = min(bounds), max(bounds)
        broke_low = any(candle.low < lower for candle in window)
        broke_high = any(candle.high > upper for candle in window)
        if not broke_low and not broke_high:
            range_status = "RANGE_CONTAINED"
        elif broke_low and broke_high:
            range_status = "RANGE_BROKE_BOTH"
        elif broke_low:
            range_status = "RANGE_BROKE_LOW"
        else:
            range_status = "RANGE_BROKE_HIGH"
        result.update(
            {
                "range_lower": lower,
                "range_upper": upper,
                "range_status": range_status,
                "direction_status": "CORRECT" if range_status == "RANGE_CONTAINED" else "WRONG",
                "target_completion_status": "NOT_APPLICABLE_RANGE",
                "invalidation_status": "NOT_APPLICABLE_RANGE",
                "first_touch_status": "NOT_APPLICABLE_RANGE",
                "full_read_status": range_status,
            }
        )
        return result
    else:
        result.update(
            {
                "direction_status": "UNSCORABLE_DIRECTION",
                "target_completion_status": "UNSCORABLE_DIRECTION",
                "invalidation_status": "UNSCORABLE_DIRECTION",
                "first_touch_status": "UNSCORABLE_DIRECTION",
                "full_read_status": "UNSCORABLE_DIRECTION",
            }
        )
        return result

    targets = _numbers(prediction.get("target_zone"))
    invalidations = _numbers(prediction.get("invalidation"))
    target = None
    if targets:
        target = min(targets) if direction in _LONG_DIRECTIONS else max(targets)
    invalidation = None
    if invalidations:
        # A zone is first invalidated at its nearest directional boundary.
        invalidation = (
            max(invalidations)
            if direction in _LONG_DIRECTIONS
            else min(invalidations)
        )
    target_at, invalidation_at, touch_order = _touches(
        window,
        direction=direction,
        target=target,
        invalidation=invalidation,
    )
    result.update(
        {
            "target_price": target,
            "invalidation_price": invalidation,
            "target_completion_status": (
                "UNSCORABLE_NO_TARGET" if target is None else ("TOUCHED" if target_at else "NOT_TOUCHED")
            ),
            "invalidation_status": (
                "UNSCORABLE_NO_INVALIDATION"
                if invalidation is None
                else ("TOUCHED" if invalidation_at else "NOT_TOUCHED")
            ),
            "target_touched_candle_start_utc": _iso(target_at) if target_at else None,
            "invalidation_touched_candle_start_utc": _iso(invalidation_at) if invalidation_at else None,
            "first_touch_status": touch_order,
        }
    )
    if target is None:
        full_status = "UNSCORABLE_NO_TARGET"
    elif invalidation is None:
        full_status = "UNSCORABLE_NO_INVALIDATION"
    elif touch_order in {"TARGET_FIRST", "TARGET_ONLY"}:
        full_status = touch_order
    elif touch_order in {"INVALIDATION_FIRST", "INVALIDATION_ONLY", "AMBIGUOUS_SAME_CANDLE"}:
        full_status = touch_order
    elif result["direction_status"] == "CORRECT":
        full_status = "DIRECTION_CORRECT_TARGET_INCOMPLETE"
    else:
        full_status = "DIRECTION_WRONG_TARGET_INCOMPLETE"
    result["full_read_status"] = full_status
    return result


def _apply_v2_scores(
    row: dict[str, Any],
    *,
    pair_charts: Mapping[str, Any] | None,
    now: datetime,
) -> None:
    pair = str(row.get("pair") or "")
    candles, candle_issue = _pair_m5_candles(pair_charts, pair)
    horizon_results = row.get("horizon_results") if isinstance(row.get("horizon_results"), dict) else {}
    for horizon, minutes in HORIZONS_MINUTES:
        current = horizon_results.get(horizon) if isinstance(horizon_results.get(horizon), dict) else {}
        if current.get("resolution_status") == "RESOLVED_MID_CANDLE_DIAGNOSTIC":
            continue
        horizon_results[horizon] = _score_horizon(
            row,
            horizon=horizon,
            minutes=minutes,
            candles=candles,
            candle_issue=candle_issue,
            now=now,
        )
    row["horizon_results"] = horizon_results
    result_30m = horizon_results.get("30m", {})
    result_2h = horizon_results.get("2h", {})
    row["actual_30m_price"] = result_30m.get("actual_price")
    row["actual_2h_price"] = result_2h.get("actual_price")
    row["actual_30m_observed_at_utc"] = result_30m.get("endpoint_observed_at_utc")
    row["actual_2h_observed_at_utc"] = result_2h.get("endpoint_observed_at_utc")
    row["thirty_minute_verdict"] = result_30m.get("full_read_status", "UNRESOLVED")
    row["two_hour_verdict"] = result_2h.get("full_read_status", "UNRESOLVED")
    statuses = [row["thirty_minute_verdict"], row["two_hour_verdict"]]
    if "UNRESOLVED" in statuses:
        row["verdict"] = "UNRESOLVED"
    elif all(status in _FULL_READ_SUCCESS for status in statuses):
        row["verdict"] = "FULL_READ_COMPLETE"
    else:
        row["verdict"] = "FULL_READ_INCOMPLETE"
    _sync_reaction_resolution(row)


def _ensure_reaction_chain(row: dict[str, Any]) -> dict[str, Any]:
    prediction_id = str(row.get("prediction_id") or "")
    chain = row.get("reaction_chain")
    if not isinstance(chain, dict):
        chain = _empty_reaction_chain(prediction_id)
        row["reaction_chain"] = chain
        return chain
    template = _empty_reaction_chain(prediction_id)
    chain.setdefault("contract", MARKET_READ_REACTION_CONTRACT)
    chain.setdefault("prediction_id", prediction_id)
    for key in ("prediction_resolution", "first_subsequent_decision", "execution_attribution", "realized_outcome"):
        if not isinstance(chain.get(key), dict):
            chain[key] = template[key]
    return chain


def _sync_reaction_resolution(row: dict[str, Any]) -> None:
    chain = _ensure_reaction_chain(row)
    horizon_results = row.get("horizon_results") if isinstance(row.get("horizon_results"), Mapping) else {}
    horizons: dict[str, Any] = {}
    resolved_count = 0
    for horizon, _minutes in HORIZONS_MINUTES:
        result = horizon_results.get(horizon) if isinstance(horizon_results.get(horizon), Mapping) else {}
        status = str(result.get("resolution_status") or "UNRESOLVED")
        if status == "RESOLVED_MID_CANDLE_DIAGNOSTIC":
            resolved_count += 1
        horizons[horizon] = {
            "status": status,
            "observed_at_utc": result.get("endpoint_observed_at_utc"),
            "resolved_at_utc": result.get("resolved_at_utc"),
            "resolution_lag_seconds": result.get("resolution_lag_seconds"),
            "direction_status": result.get("direction_status", "UNRESOLVED"),
            "target_completion_status": result.get("target_completion_status", "UNRESOLVED"),
            "invalidation_status": result.get("invalidation_status", "UNRESOLVED"),
            "first_touch_status": result.get("first_touch_status", "UNRESOLVED"),
            "full_read_status": result.get("full_read_status", "UNRESOLVED"),
            "truth_source": result.get("truth_source", MARKET_READ_TRUTH_SOURCE),
        }
    if resolved_count == len(HORIZONS_MINUTES):
        status = "RESOLVED"
    elif resolved_count:
        status = "PARTIALLY_RESOLVED"
    else:
        status = "UNRESOLVED"
    chain["prediction_resolution"] = {
        "status": status,
        "score_eligible": row.get("score_eligible") is True,
        "horizons": horizons,
    }


def _decision_receipt_payload(
    decision: Any,
    packet: Mapping[str, Any],
    *,
    status: str,
    issues: Iterable[Any],
    now: datetime,
    current_prediction_id: str | None,
) -> dict[str, Any]:
    generated = _parse_utc(_decision_value(decision, "generated_at_utc"))
    selected_lane_ids = list(_decision_value(decision, "selected_lane_ids", ()) or ())
    cancel_order_ids = list(_decision_value(decision, "cancel_order_ids", ()) or ())
    close_trade_ids = list(_decision_value(decision, "close_trade_ids", ()) or ())
    broker = packet.get("broker_snapshot") if isinstance(packet.get("broker_snapshot"), Mapping) else {}
    if isinstance(decision, Mapping):
        full_decision_payload = dict(decision)
    else:
        try:
            full_decision_payload = dict(vars(decision))
        except (AttributeError, TypeError):
            full_decision_payload = {}
    receipt_content = {
        "decision_payload": full_decision_payload,
        "generated_at_utc": _iso(generated) if generated is not None else None,
        "source_snapshot_at_utc": broker.get("fetched_at_utc"),
        "market_read_first": _decision_value(decision, "market_read_first", {}),
        "action": str(_decision_value(decision, "action") or ""),
        "selected_lane_id": _decision_value(decision, "selected_lane_id"),
        "selected_lane_ids": selected_lane_ids,
        "cancel_order_ids": cancel_order_ids,
        "close_trade_ids": close_trade_ids,
        "confidence": _decision_value(decision, "confidence"),
        "thesis": _decision_value(decision, "thesis"),
        "method": _decision_value(decision, "method"),
        "evidence_refs": list(_decision_value(decision, "evidence_refs", ()) or ()),
        "verification_status": status,
        "verification_issue_codes": _issue_codes(issues),
    }
    return {
        "status": "RESOLVED",
        "decision_receipt_id": "gptd:" + _digest(receipt_content),
        "decision_generated_at_utc": receipt_content["generated_at_utc"],
        "decision_recorded_at_utc": _iso(now),
        "action": receipt_content["action"],
        "verification_status": status,
        "verification_issue_codes": receipt_content["verification_issue_codes"],
        "market_read_prediction_id": current_prediction_id,
        "selected_lane_id": receipt_content["selected_lane_id"],
        "selected_lane_ids": selected_lane_ids,
        "cancel_order_ids": cancel_order_ids,
        "close_trade_ids": close_trade_ids,
        "reaction_link_basis": "NEXT_MARKET_READ_LEDGER_RECEIPT_IN_STRICT_RECORD_ORDER",
    }


def _bind_originating_decision(
    row: dict[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Bind a prediction to the exact decision that created that same row."""

    decision_receipt_id = str(receipt.get("decision_receipt_id") or "").strip()
    prediction_id = str(row.get("prediction_id") or "").strip()
    row["originating_decision_receipt_id"] = decision_receipt_id or None
    row["direct_execution_attribution"] = _empty_direct_execution_attribution(
        decision_receipt_id=decision_receipt_id or None,
        prediction_id=prediction_id,
        reason=(
            "NO_EXACT_EXECUTION_LINK"
            if decision_receipt_id and prediction_id
            else "ORIGINATING_DECISION_RECEIPT_ID_NOT_BOUND"
        ),
    )
    row["direct_realized_outcome"] = _empty_direct_realized_outcome()


def _execution_attribution_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    order_ids = list(
        dict.fromkeys(str(item) for item in receipt.get("cancel_order_ids", []) or [] if str(item))
    )
    trade_ids = list(
        dict.fromkeys(str(item) for item in receipt.get("close_trade_ids", []) or [] if str(item))
    )
    has_exact_ids = bool(order_ids or trade_ids)
    return {
        "status": "PARTIALLY_ATTRIBUTED" if has_exact_ids else "UNATTRIBUTED",
        "attribution_basis": "EXPLICIT_DECISION_RECEIPT_IDS" if has_exact_ids else None,
        # The GPT receipt never contains a broker fill id.  Do not infer one
        # from lane, pair or timestamp proximity.
        "fill_ids": {"status": "UNATTRIBUTED", "ids": []},
        "order_ids": {
            "status": "ATTRIBUTED_EXPLICIT_DECISION_IDS" if order_ids else "UNATTRIBUTED",
            "ids": order_ids,
        },
        "trade_ids": {
            "status": "ATTRIBUTED_EXPLICIT_DECISION_IDS" if trade_ids else "UNATTRIBUTED",
            "ids": trade_ids,
        },
        "unattributed_reason": (
            "FILL_IDS_NOT_PRESENT_IN_DECISION_RECEIPT"
            if has_exact_ids
            else "DECISION_RECEIPT_HAS_NO_BROKER_ORDER_FILL_OR_TRADE_IDS"
        ),
    }


def _execution_links_path(
    predictions_path: Path,
    explicit_path: Path | None,
) -> Path:
    return explicit_path or predictions_path.with_name(
        DEFAULT_MARKET_READ_EXECUTION_LINKS.name
    )


def _read_execution_link_evidence(
    path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return read_execution_links(path), None
    except (DecisionExecutionLineageError, OSError) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _refresh_reaction_execution_attribution(
    row: dict[str, Any],
    *,
    execution_links: Sequence[Mapping[str, Any]],
    execution_links_path: Path,
    execution_links_error: str | None,
    now: datetime,
) -> None:
    """Attach broker ids only by the two content-addressed GPT lineage ids."""

    chain = _ensure_reaction_chain(row)
    reaction = chain.get("first_subsequent_decision")
    if not isinstance(reaction, Mapping) or reaction.get("status") != "RESOLVED":
        return
    decision_receipt_id = str(reaction.get("decision_receipt_id") or "").strip()
    prediction_id = str(reaction.get("market_read_prediction_id") or "").strip()
    if not decision_receipt_id or not prediction_id:
        return
    if execution_links_error is not None:
        chain["execution_attribution"] = {
            "status": "UNATTRIBUTED",
            "attribution_basis": None,
            "decision_receipt_id": decision_receipt_id,
            "market_read_prediction_id": prediction_id,
            "execution_links_path": str(execution_links_path),
            "execution_link_artifact_status": "INVALID",
            "execution_link_artifact_error": execution_links_error,
            "fill_ids": {"status": "UNATTRIBUTED", "ids": []},
            "order_ids": {"status": "UNATTRIBUTED", "ids": []},
            "trade_ids": {"status": "UNATTRIBUTED", "ids": []},
            "transaction_ids": {"status": "UNATTRIBUTED", "ids": []},
            "pair_or_time_inference_used": False,
            "unattributed_reason": "EXECUTION_LINK_ARTIFACT_INVALID",
        }
        chain["realized_outcome"] = {
            "status": "UNRESOLVED",
            "realized_pl_jpy": None,
            "financing_jpy": None,
            "net_realized_jpy": None,
            "trade_outcomes": [],
            "unresolved_reason": "EXECUTION_LINK_ARTIFACT_INVALID",
        }
        return

    same_decision = [
        link
        for link in execution_links
        if str(link.get("decision_receipt_id") or "") == decision_receipt_id
    ]
    exact = [
        link
        for link in same_decision
        if str(link.get("market_read_prediction_id") or "") == prediction_id
    ]
    if not exact:
        if same_decision:
            chain["execution_attribution"] = {
                "status": "UNATTRIBUTED",
                "attribution_basis": None,
                "decision_receipt_id": decision_receipt_id,
                "market_read_prediction_id": prediction_id,
                "execution_links_path": str(execution_links_path),
                "execution_link_artifact_status": "READ",
                "fill_ids": {"status": "UNATTRIBUTED", "ids": []},
                "order_ids": {"status": "UNATTRIBUTED", "ids": []},
                "trade_ids": {"status": "UNATTRIBUTED", "ids": []},
                "transaction_ids": {"status": "UNATTRIBUTED", "ids": []},
                "pair_or_time_inference_used": False,
                "unattributed_reason": "EXPLICIT_EXECUTION_LINK_PREDICTION_ID_MISMATCH",
            }
        return

    def broker_ids(name: str) -> list[str]:
        return _unique_strings(
            item
            for link in exact
            for item in (
                link.get("broker_ids", {}).get(name, [])
                if isinstance(link.get("broker_ids"), Mapping)
                and isinstance(link.get("broker_ids", {}).get(name), list)
                else []
            )
        )

    order_ids = broker_ids("order_ids")
    fill_ids = broker_ids("fill_transaction_ids")
    trade_ids = broker_ids("trade_ids")
    transaction_ids = broker_ids("transaction_ids")
    link_ids = _unique_strings(link.get("link_id") for link in exact)
    chain["execution_attribution"] = {
        "status": "PARTIALLY_ATTRIBUTED",
        "attribution_basis": "EXPLICIT_ACTUAL_GATEWAY_RESPONSE_IDS_ONLY",
        "decision_receipt_id": decision_receipt_id,
        "market_read_prediction_id": prediction_id,
        "execution_links_path": str(execution_links_path),
        "execution_link_artifact_status": "VALID",
        "execution_link_ids": link_ids,
        "fill_ids": {
            "status": "ATTRIBUTED_EXPLICIT_GATEWAY_RESPONSE" if fill_ids else "UNATTRIBUTED",
            "ids": fill_ids,
        },
        "order_ids": {
            "status": "ATTRIBUTED_EXPLICIT_GATEWAY_RESPONSE" if order_ids else "UNATTRIBUTED",
            "ids": order_ids,
        },
        "trade_ids": {
            "status": "ATTRIBUTED_EXPLICIT_GATEWAY_RESPONSE" if trade_ids else "UNATTRIBUTED",
            "ids": trade_ids,
        },
        "transaction_ids": {
            "status": "ATTRIBUTED_EXPLICIT_GATEWAY_RESPONSE" if transaction_ids else "UNATTRIBUTED",
            "ids": transaction_ids,
        },
        "pair_or_time_inference_used": False,
        "resolved_at_utc": _iso(now),
        "unattributed_reason": (
            None if trade_ids else "EXPLICIT_TRADE_ID_NOT_PRESENT_IN_GATEWAY_RESPONSE_YET"
        ),
    }
    chain["realized_outcome"] = {
        "status": "UNRESOLVED",
        "realized_pl_jpy": None,
        "financing_jpy": None,
        "net_realized_jpy": None,
        "trade_outcomes": [],
        "unresolved_reason": (
            "EXECUTION_LEDGER_NOT_REFRESHED"
            if trade_ids
            else "NO_EXACT_TRADE_IDS"
        ),
    }


def _refresh_direct_execution_attribution(
    row: dict[str, Any],
    *,
    execution_links: Sequence[Mapping[str, Any]],
    execution_links_path: Path,
    execution_links_error: str | None,
    now: datetime,
) -> None:
    """Join this prediction to its own execution by exact gptd+mr2 only."""

    prediction_id = str(row.get("prediction_id") or "").strip()
    decision_receipt_id = str(
        row.get("originating_decision_receipt_id") or ""
    ).strip()
    row["originating_decision_receipt_id"] = decision_receipt_id or None
    if not prediction_id or not decision_receipt_id:
        row["direct_execution_attribution"] = _empty_direct_execution_attribution(
            decision_receipt_id=decision_receipt_id or None,
            prediction_id=prediction_id,
            reason="ORIGINATING_DECISION_RECEIPT_ID_NOT_BOUND",
        )
        row["direct_realized_outcome"] = _empty_direct_realized_outcome(
            "ORIGINATING_DECISION_RECEIPT_ID_NOT_BOUND"
        )
        return

    # Reuse the same exact-ID resolver as the reaction chain, but supply the
    # row's originating receipt rather than a subsequent receipt. The
    # synthetic chain never leaves this function and cannot introduce a
    # pair/time join.
    synthetic = {
        "prediction_id": prediction_id,
        "reaction_chain": _empty_reaction_chain(prediction_id),
    }
    synthetic_chain = synthetic["reaction_chain"]
    synthetic_chain["first_subsequent_decision"] = {
        "status": "RESOLVED",
        "decision_receipt_id": decision_receipt_id,
        "market_read_prediction_id": prediction_id,
    }
    synthetic_chain["execution_attribution"] = _empty_direct_execution_attribution(
        decision_receipt_id=decision_receipt_id,
        prediction_id=prediction_id,
    )
    synthetic_chain["realized_outcome"] = _empty_direct_realized_outcome()
    _refresh_reaction_execution_attribution(
        synthetic,
        execution_links=execution_links,
        execution_links_path=execution_links_path,
        execution_links_error=execution_links_error,
        now=now,
    )
    row["direct_execution_attribution"] = synthetic_chain["execution_attribution"]
    row["direct_realized_outcome"] = synthetic_chain["realized_outcome"]


def _attach_first_subsequent_decision(
    rows: Sequence[dict[str, Any]],
    decision: Any,
    packet: Mapping[str, Any],
    *,
    status: str,
    issues: Iterable[Any],
    now: datetime,
    current_prediction_id: str | None,
) -> None:
    receipt = _decision_receipt_payload(
        decision,
        packet,
        status=status,
        issues=issues,
        now=now,
        current_prediction_id=current_prediction_id,
    )
    for row in rows:
        chain = _ensure_reaction_chain(row)
        current = chain.get("first_subsequent_decision")
        if isinstance(current, Mapping) and current.get("status") == "RESOLVED":
            continue
        recorded_at = _parse_utc(row.get("recorded_at_utc"))
        # Without a strict, durable record-order relation this is not a safe
        # subsequent-receipt join.  Leave it explicitly unresolved.
        if recorded_at is None or now <= recorded_at:
            continue
        linked = dict(receipt)
        linked["reaction_lag_from_prediction_record_seconds"] = (
            now - recorded_at
        ).total_seconds()
        chain["first_subsequent_decision"] = linked
        chain["execution_attribution"] = _execution_attribution_from_receipt(linked)
        trade_ids = chain["execution_attribution"]["trade_ids"]["ids"]
        chain["realized_outcome"] = {
            "status": "UNRESOLVED",
            "realized_pl_jpy": None,
            "financing_jpy": None,
            "net_realized_jpy": None,
            "trade_outcomes": [],
            "unresolved_reason": (
                "EXECUTION_LEDGER_NOT_REFRESHED" if trade_ids else "NO_EXACT_TRADE_IDS"
            ),
        }


def _exact_order_fill_evidence(
    path: Path,
    order_ids: Sequence[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Read an exact ORDER_FILLED order-id mapping without market inference."""

    if not path.exists():
        return None, "EXECUTION_LEDGER_MISSING"
    requested = _unique_strings(order_ids)
    if not requested:
        return None, "EXACT_GATEWAY_ORDER_ID_MISSING"
    placeholders = ",".join("?" for _item in requested)
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(execution_events)")
            }
            required = {
                "event_uid",
                "ts_utc",
                "event_type",
                "order_id",
                "trade_id",
                "realized_pl_jpy",
                "financing_jpy",
            }
            if not required.issubset(columns):
                return None, "EXECUTION_LEDGER_SCHEMA_INVALID"
            transaction_expr = (
                "oanda_transaction_id"
                if "oanda_transaction_id" in columns
                else "NULL AS oanda_transaction_id"
            )
            rows = conn.execute(
                f"""
                SELECT event_uid, order_id, trade_id, {transaction_expr}
                FROM execution_events
                WHERE event_type = 'ORDER_FILLED'
                  AND order_id IN ({placeholders})
                ORDER BY event_uid
                """,
                tuple(requested),
            ).fetchall()
    except (OSError, sqlite3.Error, ValueError):
        return None, "EXECUTION_LEDGER_UNAVAILABLE"

    exact_rows = [
        row for row in rows if str(row["order_id"] or "").strip() in requested
    ]
    if not exact_rows:
        return None, "EXACT_GATEWAY_ORDER_ID_NOT_FILLED"
    trade_ids = _unique_strings(row["trade_id"] for row in exact_rows)
    if not trade_ids:
        return None, "EXACT_ORDER_FILL_HAS_NO_TRADE_ID"
    fill_transaction_ids = _unique_strings(
        row["oanda_transaction_id"] for row in exact_rows
    )
    return {
        "source_order_ids": requested,
        "matched_order_ids": _unique_strings(row["order_id"] for row in exact_rows),
        "trade_ids": trade_ids,
        "fill_transaction_ids": fill_transaction_ids,
        "execution_event_ids": _unique_strings(row["event_uid"] for row in exact_rows),
    }, None


def _reconcile_attributed_gateway_orders(
    attribution: dict[str, Any],
    *,
    execution_ledger_path: Path | None,
    now: datetime,
) -> str | None:
    """Enrich an exact gateway order id from exact ORDER_FILLED ledger rows."""

    if attribution.get("execution_link_artifact_status") != "VALID":
        return None
    trade_attribution = attribution.get("trade_ids")
    existing_trade_ids = (
        _unique_strings(trade_attribution.get("ids", []))
        if isinstance(trade_attribution, Mapping)
        else []
    )
    if existing_trade_ids:
        return None
    order_attribution = attribution.get("order_ids")
    order_ids = (
        _unique_strings(order_attribution.get("ids", []))
        if isinstance(order_attribution, Mapping)
        else []
    )
    if not order_ids:
        return None

    if execution_ledger_path is None:
        evidence = None
        error = "EXECUTION_LEDGER_PATH_NOT_CONFIGURED"
    else:
        evidence, error = _exact_order_fill_evidence(
            execution_ledger_path,
            order_ids,
        )
    if evidence is None:
        attribution["order_fill_reconciliation"] = {
            "status": "INVALID" if error in {
                "EXECUTION_LEDGER_MISSING",
                "EXECUTION_LEDGER_SCHEMA_INVALID",
                "EXECUTION_LEDGER_UNAVAILABLE",
                "EXECUTION_LEDGER_PATH_NOT_CONFIGURED",
            } else "UNRESOLVED",
            "attribution_basis": "EXACT_ORDER_ID_EQUALITY_ON_ORDER_FILLED",
            "source_order_ids": order_ids,
            "matched_order_ids": [],
            "execution_event_ids": [],
            "fill_transaction_ids": [],
            "trade_ids": [],
            "execution_ledger_path": (
                str(execution_ledger_path) if execution_ledger_path is not None else None
            ),
            "pair_or_time_inference_used": False,
            "error": error,
        }
        attribution["unattributed_reason"] = error
        return error

    fill_attribution = attribution.get("fill_ids")
    existing_fill_ids = (
        _unique_strings(fill_attribution.get("ids", []))
        if isinstance(fill_attribution, Mapping)
        else []
    )
    ledger_fill_ids = _unique_strings(evidence["fill_transaction_ids"])
    fill_transaction_ids = _unique_strings([*existing_fill_ids, *ledger_fill_ids])
    reconciled_trade_ids = _unique_strings(evidence["trade_ids"])
    transaction_attribution = attribution.get("transaction_ids")
    transaction_ids = (
        _unique_strings(transaction_attribution.get("ids", []))
        if isinstance(transaction_attribution, Mapping)
        else []
    )
    if existing_fill_ids and ledger_fill_ids:
        fill_status = (
            "ATTRIBUTED_EXPLICIT_GATEWAY_AND_EXACT_EXECUTION_LEDGER_ORDER_FILL"
        )
    elif ledger_fill_ids:
        fill_status = "ATTRIBUTED_EXACT_EXECUTION_LEDGER_ORDER_FILL"
    elif existing_fill_ids and isinstance(fill_attribution, Mapping):
        fill_status = str(fill_attribution.get("status") or "UNATTRIBUTED")
    else:
        fill_status = "UNATTRIBUTED"
    if transaction_ids and ledger_fill_ids:
        transaction_status = (
            "ATTRIBUTED_EXPLICIT_GATEWAY_AND_EXACT_EXECUTION_LEDGER_ORDER_FILL"
        )
    elif ledger_fill_ids:
        transaction_status = "ATTRIBUTED_EXACT_EXECUTION_LEDGER_ORDER_FILL"
    elif transaction_ids and isinstance(transaction_attribution, Mapping):
        transaction_status = str(
            transaction_attribution.get("status") or "UNATTRIBUTED"
        )
    else:
        transaction_status = "UNATTRIBUTED"
    attribution["fill_ids"] = {
        "status": fill_status,
        "ids": fill_transaction_ids,
    }
    attribution["trade_ids"] = {
        "status": "ATTRIBUTED_EXACT_EXECUTION_LEDGER_ORDER_FILL",
        "ids": reconciled_trade_ids,
    }
    attribution["transaction_ids"] = {
        "status": transaction_status,
        "ids": _unique_strings([*transaction_ids, *fill_transaction_ids]),
    }
    attribution["attribution_basis"] = (
        "EXPLICIT_ACTUAL_GATEWAY_ORDER_ID_THEN_EXACT_EXECUTION_LEDGER_ORDER_FILLED"
    )
    attribution["order_fill_reconciliation"] = {
        "status": "RESOLVED",
        "attribution_basis": "EXACT_ORDER_ID_EQUALITY_ON_ORDER_FILLED",
        **evidence,
        "execution_ledger_path": str(execution_ledger_path),
        "pair_or_time_inference_used": False,
        "resolved_at_utc": _iso(now),
        "error": None,
    }
    attribution["unattributed_reason"] = None
    return None


def _execution_trade_outcomes(
    path: Path,
    trade_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    placeholders = ",".join("?" for _item in trade_ids)
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(execution_events)")}
        required = {
            "event_uid",
            "ts_utc",
            "event_type",
            "trade_id",
            "realized_pl_jpy",
            "financing_jpy",
        }
        if not required.issubset(columns):
            return []
        rows = conn.execute(
            f"""
            SELECT event_uid, ts_utc, event_type, trade_id,
                   realized_pl_jpy, financing_jpy
            FROM execution_events
            WHERE trade_id IN ({placeholders})
              AND event_type IN ('TRADE_REDUCED', 'TRADE_CLOSED')
            ORDER BY ts_utc, event_uid
            """,
            tuple(trade_ids),
        ).fetchall()
    outcomes: list[dict[str, Any]] = []
    for trade_id in trade_ids:
        trade_rows = [row for row in rows if str(row["trade_id"] or "") == trade_id]
        closed_rows = [row for row in trade_rows if row["event_type"] == "TRADE_CLOSED"]
        realized_values = [float(row["realized_pl_jpy"]) for row in trade_rows if row["realized_pl_jpy"] is not None]
        financing_values = [float(row["financing_jpy"]) for row in trade_rows if row["financing_jpy"] is not None]
        close_pl_complete = bool(closed_rows) and all(
            row["realized_pl_jpy"] is not None for row in closed_rows
        )
        if close_pl_complete:
            outcome_status = "RESOLVED"
            unresolved_reason = None
        elif realized_values or financing_values:
            outcome_status = "PARTIALLY_RESOLVED"
            unresolved_reason = "TRADE_CLOSE_PNL_NOT_YET_COMPLETE"
        else:
            outcome_status = "UNRESOLVED"
            unresolved_reason = "NO_REALIZED_EXECUTION_EVENT"
        realized = sum(realized_values) if realized_values else None
        financing = sum(financing_values) if financing_values else None
        outcomes.append(
            {
                "trade_id": trade_id,
                "status": outcome_status,
                "realized_pl_jpy": (
                    round(realized, DIAGNOSTIC_DECIMAL_PLACES) if realized is not None else None
                ),
                "financing_jpy": (
                    round(financing, DIAGNOSTIC_DECIMAL_PLACES) if financing is not None else None
                ),
                "net_realized_jpy": (
                    round(
                        (realized or 0.0) + (financing or 0.0),
                        DIAGNOSTIC_DECIMAL_PLACES,
                    )
                    if realized is not None or financing is not None
                    else None
                ),
                "resolved_at_utc": str(closed_rows[-1]["ts_utc"]) if closed_rows else None,
                "execution_event_ids": [str(row["event_uid"]) for row in trade_rows],
                "unresolved_reason": unresolved_reason,
            }
        )
    return outcomes


def _refresh_reaction_realized_outcome(
    row: dict[str, Any],
    *,
    execution_ledger_path: Path | None,
    now: datetime,
) -> None:
    chain = _ensure_reaction_chain(row)
    attribution = chain.get("execution_attribution")
    if not isinstance(attribution, dict):
        return
    reconciliation_error = _reconcile_attributed_gateway_orders(
        attribution,
        execution_ledger_path=execution_ledger_path,
        now=now,
    )
    if reconciliation_error is not None:
        chain["realized_outcome"]["unresolved_reason"] = reconciliation_error
    trade_attribution = attribution.get("trade_ids") if isinstance(attribution, Mapping) else None
    trade_ids = (
        [str(item) for item in trade_attribution.get("ids", []) or [] if str(item)]
        if isinstance(trade_attribution, Mapping)
        else []
    )
    if not trade_ids:
        return
    if execution_ledger_path is None:
        chain["realized_outcome"]["unresolved_reason"] = "EXECUTION_LEDGER_PATH_NOT_CONFIGURED"
        return
    if not execution_ledger_path.exists():
        chain["realized_outcome"]["unresolved_reason"] = "EXECUTION_LEDGER_MISSING"
        return
    try:
        outcomes = _execution_trade_outcomes(execution_ledger_path, trade_ids)
    except (OSError, sqlite3.Error, ValueError):
        chain["realized_outcome"]["unresolved_reason"] = "EXECUTION_LEDGER_UNAVAILABLE"
        return
    if not outcomes:
        chain["realized_outcome"]["unresolved_reason"] = "NO_EXACT_TRADE_OUTCOMES"
        return
    if all(outcome["status"] == "RESOLVED" for outcome in outcomes):
        status = "RESOLVED"
        unresolved_reason = None
    elif any(outcome["status"] in {"RESOLVED", "PARTIALLY_RESOLVED"} for outcome in outcomes):
        status = "PARTIALLY_RESOLVED"
        unresolved_reason = "ONE_OR_MORE_TRADE_OUTCOMES_UNRESOLVED"
    else:
        status = "UNRESOLVED"
        unresolved_reason = "NO_RESOLVED_TRADE_OUTCOME"
    realized_values = [outcome["realized_pl_jpy"] for outcome in outcomes if outcome["realized_pl_jpy"] is not None]
    financing_values = [outcome["financing_jpy"] for outcome in outcomes if outcome["financing_jpy"] is not None]
    net_values = [outcome["net_realized_jpy"] for outcome in outcomes if outcome["net_realized_jpy"] is not None]
    chain["realized_outcome"] = {
        "status": status,
        "realized_pl_jpy": (
            round(sum(realized_values), DIAGNOSTIC_DECIMAL_PLACES) if realized_values else None
        ),
        "financing_jpy": (
            round(sum(financing_values), DIAGNOSTIC_DECIMAL_PLACES) if financing_values else None
        ),
        "net_realized_jpy": (
            round(sum(net_values), DIAGNOSTIC_DECIMAL_PLACES) if net_values else None
        ),
        "trade_outcomes": outcomes,
        "updated_at_utc": _iso(now),
        "unresolved_reason": unresolved_reason,
    }


def _refresh_direct_realized_outcome(
    row: dict[str, Any],
    *,
    execution_ledger_path: Path | None,
    now: datetime,
) -> None:
    """Resolve P/L only from trade ids in the row's direct exact-ID link."""

    prediction_id = str(row.get("prediction_id") or "")
    attribution = row.get("direct_execution_attribution")
    outcome = row.get("direct_realized_outcome")
    synthetic = {
        "prediction_id": prediction_id,
        "reaction_chain": _empty_reaction_chain(prediction_id),
    }
    synthetic_chain = synthetic["reaction_chain"]
    synthetic_chain["execution_attribution"] = (
        dict(attribution) if isinstance(attribution, Mapping) else {}
    )
    synthetic_chain["realized_outcome"] = (
        dict(outcome)
        if isinstance(outcome, Mapping)
        else _empty_direct_realized_outcome()
    )
    _refresh_reaction_realized_outcome(
        synthetic,
        execution_ledger_path=execution_ledger_path,
        now=now,
    )
    row["direct_execution_attribution"] = synthetic_chain["execution_attribution"]
    row["direct_realized_outcome"] = synthetic_chain["realized_outcome"]


def _mark_conflict(rows: Sequence[dict[str, Any]], conflict_id: str) -> None:
    for row in rows:
        row["source_snapshot_conflict"] = True
        row["source_snapshot_conflict_group_id"] = conflict_id
        reasons = _eligibility_reasons(row)
        row["score_eligible"] = False
        row["score_ineligible_reasons"] = reasons
        _sync_reaction_resolution(row)


def _coalesce_duplicate(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    existing_observations = existing.get("decision_observations")
    if not isinstance(existing_observations, list):
        existing_observations = []
    for observation in incoming.get("decision_observations", []) or []:
        if observation not in existing_observations:
            existing_observations.append(observation)
    existing["decision_observations"] = existing_observations
    existing["duplicate_observation_count"] = max(0, len(existing_observations) - 1)
    existing["last_observed_at_utc"] = incoming.get("recorded_at_utc")


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(
        numerator / denominator * 100.0,
        REPORT_PERCENT_DECIMAL_PLACES,
    )


def _normalized_score_ineligible_reasons(row: Mapping[str, Any]) -> list[str]:
    """Return bounded diagnostic reasons from an untrusted ledger row."""

    raw = row.get("score_ineligible_reasons")
    if raw is None or raw == []:
        return ["UNSPECIFIED_SCORE_INELIGIBLE"]
    if not isinstance(raw, list):
        return ["MALFORMED_SCORE_INELIGIBLE_REASONS"]

    reasons: list[str] = []
    malformed = len(raw) > 32
    for reason in raw[:32]:
        if not isinstance(reason, str):
            malformed = True
            continue
        text = reason.strip()
        if not text or len(text) > 256:
            malformed = True
            continue
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            malformed = True
            continue
        if text not in reasons:
            reasons.append(text)
    if malformed:
        reasons.append("MALFORMED_SCORE_INELIGIBLE_REASONS")
    return reasons or ["UNSPECIFIED_SCORE_INELIGIBLE"]


def _nonnegative_int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (OverflowError, TypeError, ValueError):
        return 0


def _v2_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("score_eligible") is True]
    ineligible = [row for row in rows if row.get("score_eligible") is not True]
    ineligible_reason_counts: dict[str, int] = {}
    for row in ineligible:
        for reason in _normalized_score_ineligible_reasons(row):
            ineligible_reason_counts[reason] = ineligible_reason_counts.get(reason, 0) + 1
    horizons: dict[str, Any] = {}
    for horizon, _minutes in HORIZONS_MINUTES:
        def horizon_results_for(
            source_rows: Sequence[Mapping[str, Any]],
        ) -> list[Mapping[str, Any]]:
            results: list[Mapping[str, Any]] = []
            for row in source_rows:
                horizon_results = (
                    row.get("horizon_results")
                    if isinstance(row.get("horizon_results"), Mapping)
                    else {}
                )
                result = horizon_results.get(horizon)
                results.append(result if isinstance(result, Mapping) else {})
            return results

        all_results = horizon_results_for(rows)
        eligible_results = horizon_results_for(eligible)
        ineligible_results = horizon_results_for(ineligible)

        def resolved_results(
            results: Sequence[Mapping[str, Any]],
        ) -> list[Mapping[str, Any]]:
            return [
                result
                for result in results
                if result.get("resolution_status")
                == "RESOLVED_MID_CANDLE_DIAGNOSTIC"
            ]

        resolved = resolved_results(all_results)
        eligible_resolved = resolved_results(eligible_results)
        ineligible_resolved = resolved_results(ineligible_results)
        direction = [
            result
            for result in eligible_resolved
            if result.get("direction_status") in {"CORRECT", "WRONG"}
        ]
        target = [
            result
            for result in eligible_resolved
            if result.get("target_completion_status") in {"TOUCHED", "NOT_TOUCHED"}
        ]
        full = [
            result
            for result in eligible_resolved
            if str(result.get("full_read_status") or "").startswith("UNSCORABLE") is False
        ]
        horizons[horizon] = {
            "resolved": len(resolved),
            "unresolved": len(all_results) - len(resolved),
            "eligible_resolved": len(eligible_resolved),
            "eligible_unresolved": len(eligible_results) - len(eligible_resolved),
            "ineligible_resolved": len(ineligible_resolved),
            "ineligible_unresolved": len(ineligible_results) - len(ineligible_resolved),
            "direction_scoreable": len(direction),
            "direction_correct": sum(result.get("direction_status") == "CORRECT" for result in direction),
            "direction_accuracy_pct": _pct(
                sum(result.get("direction_status") == "CORRECT" for result in direction),
                len(direction),
            ),
            "target_scoreable": len(target),
            "target_touched": sum(result.get("target_completion_status") == "TOUCHED" for result in target),
            "target_completion_pct": _pct(
                sum(result.get("target_completion_status") == "TOUCHED" for result in target),
                len(target),
            ),
            "full_read_scoreable": len(full),
            "full_read_complete": sum(result.get("full_read_status") in _FULL_READ_SUCCESS for result in full),
            "full_read_completion_pct": _pct(
                sum(result.get("full_read_status") in _FULL_READ_SUCCESS for result in full),
                len(full),
            ),
        }
    return {
        "rows": len(rows),
        "score_eligible": len(eligible),
        "score_ineligible": len(ineligible),
        "score_ineligible_reason_counts": dict(sorted(ineligible_reason_counts.items())),
        "source_snapshot_conflicts": sum(bool(row.get("source_snapshot_conflict")) for row in rows),
        "coalesced_duplicate_observations": sum(
            _nonnegative_int_or_zero(row.get("duplicate_observation_count"))
            for row in rows
        ),
        "horizons": horizons,
        "direct_execution": _direct_execution_metrics(rows),
        "reaction_chains": _reaction_metrics(rows),
    }


def _direct_execution_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attribution_statuses = [
        str(row.get("direct_execution_attribution", {}).get("status") or "UNATTRIBUTED")
        for row in rows
        if isinstance(row.get("direct_execution_attribution"), Mapping)
    ]
    realized_statuses = [
        str(row.get("direct_realized_outcome", {}).get("status") or "UNRESOLVED")
        for row in rows
        if isinstance(row.get("direct_realized_outcome"), Mapping)
    ]
    return {
        "originating_decision_bound": sum(
            bool(str(row.get("originating_decision_receipt_id") or ""))
            for row in rows
        ),
        "originating_decision_unbound": sum(
            not bool(str(row.get("originating_decision_receipt_id") or ""))
            for row in rows
        ),
        "execution_partially_attributed": sum(
            status == "PARTIALLY_ATTRIBUTED" for status in attribution_statuses
        ),
        "execution_unattributed": sum(
            status == "UNATTRIBUTED" for status in attribution_statuses
        ),
        "realized_outcome_resolved": sum(
            status == "RESOLVED" for status in realized_statuses
        ),
        "realized_outcome_partially_resolved": sum(
            status == "PARTIALLY_RESOLVED" for status in realized_statuses
        ),
        "realized_outcome_unresolved": sum(
            status == "UNRESOLVED" for status in realized_statuses
        ),
    }


def _reaction_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows_with_chain = sum(isinstance(row.get("reaction_chain"), Mapping) for row in rows)
    chains = [
        row.get("reaction_chain") if isinstance(row.get("reaction_chain"), Mapping) else {}
        for row in rows
    ]
    decision_statuses = [
        str(chain.get("first_subsequent_decision", {}).get("status") or "UNRESOLVED")
        for chain in chains
        if isinstance(chain.get("first_subsequent_decision"), Mapping)
    ]
    attribution_statuses = [
        str(chain.get("execution_attribution", {}).get("status") or "UNATTRIBUTED")
        for chain in chains
        if isinstance(chain.get("execution_attribution"), Mapping)
    ]
    realized_statuses = [
        str(chain.get("realized_outcome", {}).get("status") or "UNRESOLVED")
        for chain in chains
        if isinstance(chain.get("realized_outcome"), Mapping)
    ]
    return {
        "rows_with_chain": rows_with_chain,
        "first_subsequent_decision_resolved": sum(status == "RESOLVED" for status in decision_statuses),
        "first_subsequent_decision_unresolved": sum(status != "RESOLVED" for status in decision_statuses),
        "execution_partially_attributed": sum(
            status == "PARTIALLY_ATTRIBUTED" for status in attribution_statuses
        ),
        "execution_unattributed": sum(status == "UNATTRIBUTED" for status in attribution_statuses),
        "realized_outcome_resolved": sum(status == "RESOLVED" for status in realized_statuses),
        "realized_outcome_partially_resolved": sum(
            status == "PARTIALLY_RESOLVED" for status in realized_statuses
        ),
        "realized_outcome_unresolved": sum(status == "UNRESOLVED" for status in realized_statuses),
    }


def _write_score_report(report_path: Path, entries: Sequence[_DocumentEntry], *, now: datetime) -> None:
    legacy = [entry.payload for entry in entries if not entry.is_v2 and entry.payload]
    v2 = [entry.payload for entry in entries if entry.is_v2]
    metrics = _v2_metrics(v2)
    ineligible_reason_summary = ", ".join(
        f"{reason}={count}"
        for reason, count in metrics["score_ineligible_reason_counts"].items()
    ) or "none"
    legacy_resolved = [row for row in legacy if str(row.get("verdict") or "PENDING") != "PENDING"]
    legacy_correct = sum(row.get("verdict") == "CORRECT" for row in legacy_resolved)
    lines = [
        "# Market Read Score Report",
        "",
        f"- Generated at UTC: `{_iso(now)}`",
        f"- Truth source: `{MARKET_READ_TRUTH_SOURCE}`",
        "- Read only: `true`",
        "- Live permission: `false`",
        f"- Legacy schema-v1 rows: `{len(legacy)}` (stored verdicts preserved; not direction accuracy)",
        f"- Legacy resolved/full-correct: `{len(legacy_resolved)}` / `{legacy_correct}`",
        f"- Schema-v2 predictions: `{metrics['rows']}`",
        f"- Schema-v2 score eligible/ineligible: `{metrics['score_eligible']}` / `{metrics['score_ineligible']}`",
        f"- Schema-v2 score-ineligible reasons: `{ineligible_reason_summary}`",
        f"- Coalesced exact duplicate observations: `{metrics['coalesced_duplicate_observations']}`",
        f"- Source-snapshot conflict rows: `{metrics['source_snapshot_conflicts']}`",
        f"- Direct originating decision bound/unbound: `{metrics['direct_execution']['originating_decision_bound']}` / `{metrics['direct_execution']['originating_decision_unbound']}`",
        f"- Direct execution partially attributed/unattributed: `{metrics['direct_execution']['execution_partially_attributed']}` / `{metrics['direct_execution']['execution_unattributed']}`",
        f"- Direct realized outcome resolved/partial/unresolved: `{metrics['direct_execution']['realized_outcome_resolved']}` / `{metrics['direct_execution']['realized_outcome_partially_resolved']}` / `{metrics['direct_execution']['realized_outcome_unresolved']}`",
        f"- First subsequent decision linked/unresolved: `{metrics['reaction_chains']['first_subsequent_decision_resolved']}` / `{metrics['reaction_chains']['first_subsequent_decision_unresolved']}`",
        f"- Reaction execution partially attributed/unattributed: `{metrics['reaction_chains']['execution_partially_attributed']}` / `{metrics['reaction_chains']['execution_unattributed']}`",
        f"- Reaction realized outcome resolved/partial/unresolved: `{metrics['reaction_chains']['realized_outcome_resolved']}` / `{metrics['reaction_chains']['realized_outcome_partially_resolved']}` / `{metrics['reaction_chains']['realized_outcome_unresolved']}`",
        "",
        "## Honest v2 Metrics",
        "",
    ]
    for horizon, _minutes in HORIZONS_MINUTES:
        item = metrics["horizons"][horizon]
        lines.extend(
            [
                f"### {horizon}",
                "",
                f"- Lifecycle resolved/unresolved (all v2): `{item['resolved']}` / `{item['unresolved']}`",
                f"- Score-eligible resolved/unresolved: `{item['eligible_resolved']}` / `{item['eligible_unresolved']}`",
                f"- Score-ineligible resolved/unresolved: `{item['ineligible_resolved']}` / `{item['ineligible_unresolved']}`",
                f"- Direction accuracy: `{item['direction_accuracy_pct']}` ({item['direction_correct']}/{item['direction_scoreable']})",
                f"- Target completion: `{item['target_completion_pct']}` ({item['target_touched']}/{item['target_scoreable']})",
                f"- Full-read completion: `{item['full_read_completion_pct']}` ({item['full_read_complete']}/{item['full_read_scoreable']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Direction accuracy is endpoint direction only; it is not target completion.",
            "- Target/invalidation first-touch is emitted only from a complete M5 path.",
            "- MID_CANDLE_DIAGNOSTIC is not bid/ask execution proof and cannot grant live permission.",
            "- Incomplete or aged-out M5 windows remain UNRESOLVED.",
            "",
            "## Recent v2 Predictions",
            "",
        ]
    )
    for row in v2[-RECENT_REPORT_PREDICTIONS_LIMIT:]:
        h30 = row.get("horizon_results", {}).get("30m", {})
        h2 = row.get("horizon_results", {}).get("2h", {})
        chain = row.get("reaction_chain") if isinstance(row.get("reaction_chain"), Mapping) else {}
        reaction = chain.get("first_subsequent_decision") if isinstance(chain.get("first_subsequent_decision"), Mapping) else {}
        attribution = chain.get("execution_attribution") if isinstance(chain.get("execution_attribution"), Mapping) else {}
        realized = chain.get("realized_outcome") if isinstance(chain.get("realized_outcome"), Mapping) else {}
        direct_attribution = (
            row.get("direct_execution_attribution")
            if isinstance(row.get("direct_execution_attribution"), Mapping)
            else {}
        )
        direct_realized = (
            row.get("direct_realized_outcome")
            if isinstance(row.get("direct_realized_outcome"), Mapping)
            else {}
        )
        lines.append(
            "- "
            f"`{row.get('generated_at_utc')}` {row.get('pair') or 'unknown'} {row.get('direction') or 'UNKNOWN'} "
            f"eligible=`{row.get('score_eligible')}` "
            f"30m(direction={h30.get('direction_status')}, target={h30.get('target_completion_status')}, full={h30.get('full_read_status')}) "
            f"2h(direction={h2.get('direction_status')}, target={h2.get('target_completion_status')}, full={h2.get('full_read_status')}) "
            f"direct(decision={row.get('originating_decision_receipt_id')}, "
            f"execution={direct_attribution.get('status', 'UNATTRIBUTED')}, "
            f"realized={direct_realized.get('status', 'UNRESOLVED')}) "
            f"reaction(decision={reaction.get('status', 'UNRESOLVED')}, action={reaction.get('action')}, "
            f"execution={attribution.get('status', 'UNATTRIBUTED')}, realized={realized.get('status', 'UNRESOLVED')})"
        )
    if not v2:
        lines.append("- none")
    _atomic_write_text(report_path, "\n".join(lines) + "\n")


def record_market_read_prediction(
    decision: Any,
    packet: Mapping[str, Any],
    *,
    status: str,
    issues: Iterable[Any],
    predictions_path: Path,
    report_path: Path,
    pair_charts_path: Path | None,
    execution_ledger_path: Path | None = None,
    execution_links_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Score due v2 rows and record/coalesce the current read.

    Any ledger-integrity failure returns a diagnostic result and leaves the
    existing bytes untouched.  It never raises into the execution path.
    """

    now = (now or _utc_now()).astimezone(timezone.utc)
    issue_items = tuple(issues)
    try:
        with _ledger_lock(predictions_path, exclusive=True):
            entries = _read_document(predictions_path)
            pair_charts = _load_pair_charts(pair_charts_path)
            existing_v2_rows = [entry.payload for entry in entries if entry.is_v2]
            for entry in entries:
                if entry.is_v2:
                    _apply_v2_scores(entry.payload, pair_charts=pair_charts, now=now)

            incoming = _build_v2_row(
                decision,
                packet,
                status=status,
                issues=issue_items,
                now=now,
            )
            current_decision_receipt = _decision_receipt_payload(
                decision,
                packet,
                status=status,
                issues=issue_items,
                now=now,
                current_prediction_id=(
                    str(incoming.get("prediction_id")) if incoming is not None else None
                ),
            )
            if incoming is not None:
                _bind_originating_decision(incoming, current_decision_receipt)
            _attach_first_subsequent_decision(
                existing_v2_rows,
                decision,
                packet,
                status=status,
                issues=issue_items,
                now=now,
                current_prediction_id=(
                    str(incoming.get("prediction_id")) if incoming is not None else None
                ),
            )
            resolved_execution_links_path = _execution_links_path(
                predictions_path,
                execution_links_path,
            )
            execution_links, execution_links_error = _read_execution_link_evidence(
                resolved_execution_links_path
            )
            for row in existing_v2_rows:
                _refresh_direct_execution_attribution(
                    row,
                    execution_links=execution_links,
                    execution_links_path=resolved_execution_links_path,
                    execution_links_error=execution_links_error,
                    now=now,
                )
                _refresh_direct_realized_outcome(
                    row,
                    execution_ledger_path=execution_ledger_path,
                    now=now,
                )
                _refresh_reaction_execution_attribution(
                    row,
                    execution_links=execution_links,
                    execution_links_path=resolved_execution_links_path,
                    execution_links_error=execution_links_error,
                    now=now,
                )
                _refresh_reaction_realized_outcome(
                    row,
                    execution_ledger_path=execution_ledger_path,
                    now=now,
                )
            if incoming is not None:
                _refresh_direct_execution_attribution(
                    incoming,
                    execution_links=execution_links,
                    execution_links_path=resolved_execution_links_path,
                    execution_links_error=execution_links_error,
                    now=now,
                )
                _refresh_direct_realized_outcome(
                    incoming,
                    execution_ledger_path=execution_ledger_path,
                    now=now,
                )
            result_status = "NO_MARKET_READ_FIRST"
            selected_row: dict[str, Any] | None = None
            if incoming is not None:
                source_id = incoming.get("source_snapshot_identity")
                same_source = [
                    entry.payload
                    for entry in entries
                    if entry.is_v2
                    and source_id
                    and entry.payload.get("source_snapshot_identity") == source_id
                ]
                exact = next(
                    (
                        row
                        for row in same_source
                        if row.get("semantic_fingerprint") == incoming.get("semantic_fingerprint")
                    ),
                    None,
                )
                if exact is not None:
                    _coalesce_duplicate(exact, incoming)
                    selected_row = exact
                    result_status = "COALESCED_EXACT_DUPLICATE"
                else:
                    if same_source:
                        conflict_id = "mrc:" + _digest(
                            {
                                "source_snapshot_identity": source_id,
                                "semantic_fingerprints": sorted(
                                    [str(row.get("semantic_fingerprint") or "") for row in same_source]
                                    + [str(incoming.get("semantic_fingerprint") or "")]
                                ),
                            }
                        )
                        _mark_conflict([*same_source, incoming], conflict_id)
                        result_status = "RECORDED_SOURCE_SNAPSHOT_CONFLICT"
                    else:
                        result_status = "RECORDED"
                    _apply_v2_scores(incoming, pair_charts=pair_charts, now=now)
                    entries.append(
                        _DocumentEntry(
                            raw="",
                            payload=incoming,
                            is_v2=True,
                        )
                    )
                    selected_row = incoming

            _atomic_write_text(predictions_path, _serialize_document(entries))
            report_error: str | None = None
            try:
                _write_score_report(report_path, entries, now=now)
            except OSError as exc:
                # The append-only evidence is already durable.  Surface a
                # retryable report failure without falsely claiming that the
                # ledger write failed or attempting to roll evidence back.
                report_error = str(exc)
            return {
                "status": result_status,
                "predictions_path": str(predictions_path),
                "report_path": str(report_path),
                "schema_version": MARKET_READ_SCHEMA_VERSION,
                "prediction_id": selected_row.get("prediction_id") if selected_row else None,
                "decision_receipt_id": current_decision_receipt.get("decision_receipt_id"),
                "decision_receipt_recorded_at_utc": current_decision_receipt.get(
                    "decision_recorded_at_utc"
                ),
                "pair": selected_row.get("pair") if selected_row else None,
                "direction": selected_row.get("direction") if selected_row else None,
                "verdict": selected_row.get("verdict") if selected_row else None,
                "truth_source": MARKET_READ_TRUTH_SOURCE,
                "read_only": True,
                "live_permission": False,
                "execution_links_path": str(resolved_execution_links_path),
                "execution_links_status": (
                    "INVALID" if execution_links_error else "VALID"
                ),
                "execution_links_error": execution_links_error,
                "report_status": "WRITE_FAILED" if report_error else "WRITTEN",
                "report_error": report_error,
            }
    except (MarketReadLedgerError, OSError) as exc:
        return {
            "status": "MARKET_READ_LEDGER_INVALID",
            "predictions_path": str(predictions_path),
            "report_path": str(report_path),
            "schema_version": MARKET_READ_SCHEMA_VERSION,
            "error": str(exc),
            "read_only": True,
            "live_permission": False,
        }


def refresh_market_read_measurements(
    *,
    predictions_path: Path,
    report_path: Path,
    pair_charts_path: Path | None,
    execution_ledger_path: Path | None = None,
    execution_links_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Resolve newly due v2 evidence before the next model decision.

    This updates diagnostic truth and exact-ID realized outcomes only.  It does
    not add a prediction, attach the current decision, call a broker, or alter
    any verifier/gateway/risk permission.
    """

    now = (now or _utc_now()).astimezone(timezone.utc)
    try:
        with _ledger_lock(predictions_path, exclusive=True):
            entries = _read_document(predictions_path)
            v2_rows = [entry.payload for entry in entries if entry.is_v2]
            before = _canonical_json(v2_rows)
            pair_charts = _load_pair_charts(pair_charts_path)
            resolved_execution_links_path = _execution_links_path(
                predictions_path,
                execution_links_path,
            )
            execution_links, execution_links_error = _read_execution_link_evidence(
                resolved_execution_links_path
            )
            for row in v2_rows:
                _apply_v2_scores(row, pair_charts=pair_charts, now=now)
                _refresh_direct_execution_attribution(
                    row,
                    execution_links=execution_links,
                    execution_links_path=resolved_execution_links_path,
                    execution_links_error=execution_links_error,
                    now=now,
                )
                _refresh_direct_realized_outcome(
                    row,
                    execution_ledger_path=execution_ledger_path,
                    now=now,
                )
                _refresh_reaction_execution_attribution(
                    row,
                    execution_links=execution_links,
                    execution_links_path=resolved_execution_links_path,
                    execution_links_error=execution_links_error,
                    now=now,
                )
                _refresh_reaction_realized_outcome(
                    row,
                    execution_ledger_path=execution_ledger_path,
                    now=now,
                )
            changed = before != _canonical_json(v2_rows)
            report_error: str | None = None
            if changed:
                _atomic_write_text(predictions_path, _serialize_document(entries))
                try:
                    _write_score_report(report_path, entries, now=now)
                except OSError as exc:
                    report_error = str(exc)
            return {
                "status": "REFRESHED" if changed else "NO_CHANGE",
                "schema_version": MARKET_READ_SCHEMA_VERSION,
                "v2_rows": len(v2_rows),
                "changed": changed,
                "truth_source": MARKET_READ_TRUTH_SOURCE,
                "read_only_measurement": True,
                "live_permission": False,
                "may_change_execution_permission": False,
                "execution_links_path": str(resolved_execution_links_path),
                "execution_links_status": (
                    "INVALID" if execution_links_error else "VALID"
                ),
                "execution_links_error": execution_links_error,
                "report_status": (
                    "WRITE_FAILED"
                    if report_error
                    else ("WRITTEN" if changed else "NOT_REQUIRED")
                ),
                "report_error": report_error,
            }
    except (MarketReadLedgerError, OSError) as exc:
        return {
            "status": "MARKET_READ_LEDGER_INVALID",
            "schema_version": MARKET_READ_SCHEMA_VERSION,
            "error": str(exc),
            "read_only_measurement": True,
            "live_permission": False,
            "may_change_execution_permission": False,
        }


def market_read_feedback_summary(path: Path) -> dict[str, Any]:
    """Return bounded v2 feedback for the next GPT decision packet.

    The packet is explicitly advisory.  No consumer may use it as a live gate,
    permission bit, risk relaxation or replacement for current broker truth.
    """

    base: dict[str, Any] = {
        "evidence_ref": "market_read:feedback",
        "schema_version": MARKET_READ_SCHEMA_VERSION,
        "read_only": True,
        "advisory_only": True,
        "live_permission": False,
        "may_change_execution_permission": False,
        "truth_source": MARKET_READ_TRUTH_SOURCE,
        "status": "NO_V2_EVIDENCE",
        "metrics": _v2_metrics([]),
        "latest_resolved": [],
    }
    try:
        with _ledger_lock(path, exclusive=False):
            entries = _read_document(path)
    except (MarketReadLedgerError, OSError) as exc:
        base["status"] = "UNAVAILABLE"
        base["error"] = str(exc)
        return base
    rows = [entry.payload for entry in entries if entry.is_v2]
    base["metrics"] = _v2_metrics(rows)
    if not rows:
        return base
    resolved_examples: list[dict[str, Any]] = []
    for row in reversed(rows):
        if row.get("score_eligible") is not True:
            continue
        horizon_results = row.get("horizon_results") if isinstance(row.get("horizon_results"), Mapping) else {}
        if not any(
            isinstance(horizon_results.get(horizon), Mapping)
            and horizon_results.get(horizon, {}).get("resolution_status") == "RESOLVED_MID_CANDLE_DIAGNOSTIC"
            for horizon, _minutes in HORIZONS_MINUTES
        ):
            continue
        chain = row.get("reaction_chain") if isinstance(row.get("reaction_chain"), Mapping) else {}
        reaction = (
            chain.get("first_subsequent_decision")
            if isinstance(chain.get("first_subsequent_decision"), Mapping)
            else {}
        )
        attribution = (
            chain.get("execution_attribution")
            if isinstance(chain.get("execution_attribution"), Mapping)
            else {}
        )
        realized = (
            chain.get("realized_outcome")
            if isinstance(chain.get("realized_outcome"), Mapping)
            else {}
        )
        direct_attribution = (
            row.get("direct_execution_attribution")
            if isinstance(row.get("direct_execution_attribution"), Mapping)
            else {}
        )
        direct_realized = (
            row.get("direct_realized_outcome")
            if isinstance(row.get("direct_realized_outcome"), Mapping)
            else {}
        )
        resolved_examples.append(
            {
                "prediction_id": row.get("prediction_id"),
                "generated_at_utc": row.get("generated_at_utc"),
                "pair": row.get("pair"),
                "direction": row.get("direction"),
                "horizons": {
                    horizon: {
                        "direction_status": horizon_results.get(horizon, {}).get("direction_status"),
                        "target_completion_status": horizon_results.get(horizon, {}).get(
                            "target_completion_status"
                        ),
                        "full_read_status": horizon_results.get(horizon, {}).get("full_read_status"),
                    }
                    for horizon, _minutes in HORIZONS_MINUTES
                },
                "direct_execution": {
                    "originating_decision_receipt_id": row.get(
                        "originating_decision_receipt_id"
                    ),
                    "execution_attribution_status": direct_attribution.get(
                        "status", "UNATTRIBUTED"
                    ),
                    "realized_outcome_status": direct_realized.get(
                        "status", "UNRESOLVED"
                    ),
                    "realized_pl_jpy": direct_realized.get("realized_pl_jpy"),
                },
                "reaction": {
                    "first_subsequent_decision_status": reaction.get("status", "UNRESOLVED"),
                    "decision_receipt_id": reaction.get("decision_receipt_id"),
                    "decision_recorded_at_utc": reaction.get("decision_recorded_at_utc"),
                    "action": reaction.get("action"),
                    "execution_attribution_status": attribution.get("status", "UNATTRIBUTED"),
                    "realized_outcome_status": realized.get("status", "UNRESOLVED"),
                    "realized_pl_jpy": realized.get("realized_pl_jpy"),
                },
            }
        )
        if len(resolved_examples) >= FEEDBACK_RESOLVED_EXAMPLES_LIMIT:
            break
    base["latest_resolved"] = resolved_examples
    base["status"] = "OK" if resolved_examples else "V2_EVIDENCE_UNRESOLVED"
    return base
