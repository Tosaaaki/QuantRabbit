#!/usr/bin/env python3
"""Exact-rational event-sourced reference engine for paper-research FX V2.

This module is intentionally a pure computation boundary.  It accepts exactly
nine raw artifact byte strings, parses and validates them independently, and
projects an event-sourced, balanced journal into canonical ledger bytes and
metrics.  It performs no filesystem, network, process, broker, credential, or
deployment operation and imports no project module.

The journal representation is deliberately different from the disposition-
first Oracle implementation.  Every economic value is derived from immutable
market/proposal events and exact-rational postings; Oracle ledger or manifest
bytes are never accepted as input.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


__all__ = (
    "ENGINE_ID",
    "ReferenceError",
    "decode_reference_input",
    "replay_reference",
)
ENGINE_ID = "EVENT_SOURCED_DOUBLE_ENTRY_REFERENCE_V1"
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
ARTIFACT_KEYS = frozenset({
    "source_blob",
    "source_manifest",
    "proposal",
    "execution_policy",
    "inventory_policy",
    "accounting_policy",
    "evaluation_policy",
    "instrument_registry",
    "authority_policy",
})
FORBIDDEN_PROPOSAL_TOKENS = frozenset({
    "signalid", "fill", "fillprice", "path", "mfe", "mae", "pnl", "cost",
    "equity", "drawdown", "dd", "cvar", "profit", "return",
})
JOURNAL_ACCOUNT_ORDER = (
    "POSITION_BASIS",
    "POSITION_CONTROL",
    "UNREALIZED_ASSET",
    "UNREALIZED_PNL",
    "COMMISSION_EXPENSE",
    "FINANCING_EXPENSE",
    "SETTLEMENT_CASH",
    "REALIZED_TRADING_PNL",
    "COMMON_GROSS_REFERENCE",
    "FILL_SIZING_DRAG",
    "LATENCY_SPREAD_SLIPPAGE_DRAG",
    "DIRECT_COST",
    "ADMISSION_OPPORTUNITY_DRAG",
    "NET_PNL_CONTROL",
)
JOURNAL_ACCOUNTS = frozenset(JOURNAL_ACCOUNT_ORDER)
ZERO_SHA256 = "0" * 64
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PAIR_PATTERN = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
RATIO_SCALE = 10**18
JPY_MICROS_PER_YEN = 1_000_000
BASE_MICROUNITS_PER_UNIT = 1_000_000
PRICE_SUBPIP_SCALE = 1_000_000
DAY_NS = 86_400_000_000_000
MAX_SOURCE_ROWS = 5_000_000
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024


class ReferenceError(RuntimeError):
    """Fail-closed reference-input or accounting error."""


@dataclass(frozen=True)
class MarketTick:
    provider_id: str
    instrument: str
    bid_ticks: int
    ask_ticks: int
    tick_scale: int
    source_ts_ns: int
    arrival_ts_ns: int
    sequence: int
    source_event_sha256: str
    source_prefix_root_sha256: str


@dataclass(frozen=True)
class Proposal:
    ordinal: int
    decision_source_ts_ns: int
    decision_arrival_ts_ns: int
    decision_source_event_sha256: str
    completed_data_watermark_source_ts_ns: int
    completed_data_prefix_root_sha256: str
    instrument: str
    direction: int
    target_notional_jpy_micros: int
    max_age_ns: int
    worker_key: str


@dataclass(frozen=True)
class ArmTerms:
    latency_ns: int
    slippage_micropips_per_side: int
    commission_ppm_per_side: int
    financing_ppm_per_day: int
    raw_mid: bool


@dataclass(frozen=True)
class Posting:
    account: str
    amount: Fraction


@dataclass(frozen=True)
class JournalTransaction:
    sequence: int
    arrival_ts_ns: int
    arm: str
    proposal_ordinal: int
    event_kind: str
    event_id: str
    source_event_sha256: str | None
    postings: tuple[Posting, ...]


@dataclass(frozen=True)
class ReferenceInput:
    ticks: tuple[MarketTick, ...]
    books: Mapping[str, tuple[MarketTick, ...]]
    proposals: tuple[Proposal, ...]
    candidate_key: str
    provenance: Mapping[str, str]
    arms: Mapping[str, ArmTerms]
    max_trade_quote_staleness_ns: int
    inventory: Mapping[str, Any]
    accounting: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    authority: Mapping[str, Any]
    registry: Mapping[str, Mapping[str, int]]
    execution_policy_sha256: str
    raw_hashes: Mapping[str, str]


@dataclass
class PositionLot:
    arm: str
    proposal: Proposal
    signal_id: str
    economic_lot_id: str
    common: Mapping[str, Any]
    entry: MarketTick
    entry_price: Fraction
    entry_price_numerator: int
    entry_price_denominator: int
    units_micros: int
    entry_notional_exact: Fraction
    entry_notional_rounded: int
    due_arrival_ns: int
    entry_commission_exact: Fraction = Fraction(0, 1)
    last_mark_pnl_exact: Fraction = Fraction(0, 1)
    last_mark_arrival_ns: int | None = None
    signed_exposure_after_entry: Mapping[str, int] | None = None
    gross_after_entry: int | None = None
    marked_equity_after_entry: int | None = None
    required_margin_after_entry: int | None = None
    free_margin_after_entry: int | None = None
    closed_disposition: ClosedDisposition | None = None


@dataclass(frozen=True)
class ClosedDisposition:
    """A settled close event from which ledger and metrics are projected."""

    position: PositionLot
    exit_tick: MarketTick
    exit_reason: str
    settlement_arrival_ns: int
    values: Mapping[str, Any]
    common_gross_jpy_micros: int
    arm_common_gross_jpy_micros: int
    fill_sizing_drag_jpy_micros: int
    execution_drag_jpy_micros: int


@dataclass(frozen=True)
class RejectedDisposition:
    """A causally settled proposal disposition with no opened position."""

    arm: str
    proposal: Proposal
    signal_id: str
    economic_lot_id: str
    reason: str
    common: Mapping[str, Any] | None
    known_arrival_ns: int
    settlement_arrival_ns: int


@dataclass(frozen=True)
class PendingRejection:
    arm: str
    proposal: Proposal
    signal_id: str
    economic_lot_id: str
    reason: str
    common: Mapping[str, Any] | None
    known_arrival_ns: int
    settlement_arrival_ns: int


@dataclass(frozen=True)
class RiskSnapshot:
    arrival_ts_ns: int
    source_watermark_ts_ns: int
    marked_equity_jpy_micros: int
    gross_notional_jpy_micros: int
    required_margin_jpy_micros: int
    free_margin_jpy_micros: int
    signed_currency_exposure_jpy_micros: Mapping[str, int]
    margin_ratio_pass: bool


Disposition = ClosedDisposition | RejectedDisposition


@dataclass(frozen=True)
class ArmReplay:
    positions: tuple[PositionLot, ...]
    risk_snapshots: tuple[RiskSnapshot, ...]
    boundary_equities: Mapping[int, int]


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReferenceError("value is not canonical JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    if type(value) is tuple:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _reject_constant(_: str) -> None:
    raise ReferenceError("non-finite JSON number forbidden")


def _parse_integer(token: str) -> int:
    if token == "-0":
        raise ReferenceError("negative zero forbidden")
    return int(token)


def _reject_float(_: str) -> None:
    raise ReferenceError("JSON fractional/exponent number forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes, label: str, *, require_lf: bool = True) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_JSON_BYTES:
        raise ReferenceError(f"{label} byte envelope invalid")
    if require_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ReferenceError(f"{label} must have one terminal LF")
        payload = raw[:-1]
    else:
        payload = raw
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceError(f"{label} JSON invalid") from error
    if type(value) is not dict or canonical_bytes(value) != payload:
        raise ReferenceError(f"{label} is not canonical object JSON")
    return value


def _exact_keys(value: Any, expected: set[str] | frozenset[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(expected):
        raise ReferenceError(f"{label} exact key set mismatch")


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ReferenceError(f"{label} must be exact integer")
    if minimum is not None and value < minimum:
        raise ReferenceError(f"{label} below minimum")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ReferenceError(f"{label} must be exact boolean")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ReferenceError(f"{label} must be nonempty text")
    return value


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ReferenceError(f"{label} must be lowercase SHA-256")
    return value


def _embedded_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_bytes(unsigned))


def _require_embedded(value: Mapping[str, Any], field: str, label: str) -> None:
    if _digest(value.get(field), f"{label}.{field}") != _embedded_hash(value, field):
        raise ReferenceError(f"{label} embedded hash mismatch")


def _pair(instrument: Any) -> tuple[str, str]:
    if type(instrument) is not str or PAIR_PATTERN.fullmatch(instrument) is None:
        raise ReferenceError("FX instrument invalid")
    base, quote = instrument.split("_", 1)
    if base == quote:
        raise ReferenceError("FX pair currencies must differ")
    return base, quote


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_producer_fields(value: Any, location: str = "proposal") -> None:
    if type(value) is dict:
        for key, item in value.items():
            normalized = _normalized_key(key)
            words = {
                word
                for word in re.split(r"[^a-z0-9]+", key.casefold())
                if word
            }
            if any(
                token in normalized if token != "dd" else token in words
                for token in FORBIDDEN_PROPOSAL_TOKENS
            ):
                raise ReferenceError(f"producer outcome field forbidden at {location}.{key}")
            _reject_producer_fields(item, f"{location}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _reject_producer_fields(item, f"{location}[{index}]")


def _month_bounds_ns(month_id: str) -> tuple[int, int]:
    if type(month_id) is not str or re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month_id) is None:
        raise ReferenceError("month id invalid")
    year = int(month_id[:4])
    month = int(month_id[5:])
    next_year, next_month = (
        (year + 1, 1) if month == 12 else (year, month + 1)
    )
    return (
        _days_from_civil(year, month, 1) * DAY_NS,
        _days_from_civil(next_year, next_month, 1) * DAY_NS,
    )


def _days_from_civil(year: int, month: int, day: int) -> int:
    """Gregorian civil date to days since 1970-01-01, with no lazy imports."""
    adjusted_year = year - (1 if month <= 2 else 0)
    era = adjusted_year // 400
    year_of_era = adjusted_year - era * 400
    month_prime = month + (-3 if month > 2 else 9)
    day_of_year = (153 * month_prime + 2) // 5 + day - 1
    day_of_era = (
        year_of_era * 365
        + year_of_era // 4
        - year_of_era // 100
        + day_of_year
    )
    return era * 146_097 + day_of_era - 719_468


def _civil_from_days(days_since_epoch: int) -> tuple[int, int, int]:
    """Inverse Gregorian conversion for UTC month grids, using integers only."""
    shifted = days_since_epoch + 719_468
    era = shifted // 146_097
    day_of_era = shifted - era * 146_097
    year_of_era = (
        day_of_era
        - day_of_era // 1_460
        + day_of_era // 36_524
        - day_of_era // 146_096
    ) // 365
    year = year_of_era + era * 400
    day_of_year = day_of_era - (
        365 * year_of_era + year_of_era // 4 - year_of_era // 100
    )
    month_prime = (5 * day_of_year + 2) // 153
    day = day_of_year - (153 * month_prime + 2) // 5 + 1
    month = month_prime + (3 if month_prime < 10 else -9)
    year += 1 if month <= 2 else 0
    return year, month, day


def _intersecting_months(start_ns: int, end_ns: int) -> list[str]:
    if start_ns >= end_ns:
        return []
    year, month, _ = _civil_from_days(start_ns // DAY_NS)
    result: list[str] = []
    while _days_from_civil(year, month, 1) * DAY_NS < end_ns:
        result.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return result


def _ratio_text(value: Fraction, *, outward_nonnegative: bool = False) -> str:
    if outward_nonnegative:
        if value < 0:
            raise ReferenceError("outward ratio must be nonnegative")
        scaled = (value.numerator * RATIO_SCALE + value.denominator - 1) // value.denominator
    else:
        scaled = value.numerator * RATIO_SCALE // value.denominator
    sign = "-" if scaled < 0 else ""
    magnitude = abs(scaled)
    return f"{sign}{magnitude // RATIO_SCALE}.{magnitude % RATIO_SCALE:018d}"


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceil_nonnegative(value: Fraction) -> int:
    if value < 0:
        raise ReferenceError("positive cost/risk cannot be negative")
    return (value.numerator + value.denominator - 1) // value.denominator


def _decode_registry(raw: bytes) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    payload = _decode_json(raw, "instrument registry")
    _exact_keys(
        payload,
        {"schema_version", "registry_id", "instruments", "registry_sha256"},
        "instrument registry",
    )
    if _integer(payload["schema_version"], "registry schema") != 1 \
            or payload["registry_id"] != "FROZEN_FX_INSTRUMENT_REGISTRY_V1":
        raise ReferenceError("instrument registry identity mismatch")
    _require_embedded(payload, "registry_sha256", "instrument registry")
    instruments = payload["instruments"]
    if type(instruments) is not dict or not instruments or list(instruments) != sorted(instruments):
        raise ReferenceError("instrument registry must be nonempty and sorted")
    result: dict[str, dict[str, int]] = {}
    economic_pairs: set[tuple[str, str]] = set()
    for instrument, spec in instruments.items():
        base, quote = _pair(instrument)
        pair_key = tuple(sorted((base, quote)))
        if pair_key in economic_pairs:
            raise ReferenceError("inverse duplicate instrument forbidden")
        economic_pairs.add(pair_key)
        _exact_keys(spec, {"price_scale", "pip_ticks"}, f"instrument {instrument}")
        scale = _integer(spec["price_scale"], f"{instrument}.price_scale", 1)
        pip_ticks = _integer(spec["pip_ticks"], f"{instrument}.pip_ticks", 1)
        if pip_ticks >= scale:
            raise ReferenceError("instrument pip convention invalid")
        result[instrument] = {"price_scale": scale, "pip_ticks": pip_ticks}
    return payload, result


def _decode_source(
    blob: bytes,
    manifest_raw: bytes,
    registry_payload: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> tuple[tuple[MarketTick, ...], dict[str, tuple[MarketTick, ...]]]:
    if type(blob) is not bytes or not blob or len(blob) > MAX_SOURCE_BYTES \
            or not blob.endswith(b"\n") or blob.endswith(b"\n\n"):
        raise ReferenceError("source blob byte envelope invalid")
    manifest = _decode_json(manifest_raw, "source manifest")
    _exact_keys(
        manifest,
        {
            "schema_version", "source_bytes_sha256", "source_size_bytes", "event_count",
            "first_source_ts_ns", "last_source_ts_ns", "provider_allowlist",
            "instrument_registry_sha256", "stream_policies", "lossless", "manifest_sha256",
        },
        "source manifest",
    )
    if _integer(manifest["schema_version"], "source manifest schema") != 2:
        raise ReferenceError("source manifest schema mismatch")
    _require_embedded(manifest, "manifest_sha256", "source manifest")
    if _digest(manifest["source_bytes_sha256"], "source bytes hash") != sha256_bytes(blob) \
            or _integer(manifest["source_size_bytes"], "source size", 0) != len(blob):
        raise ReferenceError("source manifest byte binding mismatch")
    if manifest["instrument_registry_sha256"] != registry_payload["registry_sha256"]:
        raise ReferenceError("source manifest registry binding mismatch")
    providers = manifest["provider_allowlist"]
    if type(providers) is not list or not providers or providers != sorted(set(providers)) \
            or any(type(item) is not str or not item for item in providers):
        raise ReferenceError("provider allowlist invalid")
    if _boolean(manifest["lossless"], "source lossless") is not True:
        raise ReferenceError("source must be declared lossless")
    raw_policies = manifest["stream_policies"]
    if type(raw_policies) is not list or not raw_policies:
        raise ReferenceError("stream policies missing")
    stream_policies: dict[tuple[str, str], dict[str, Any]] = {}
    policy_order: list[tuple[str, str]] = []
    for spec in raw_policies:
        _exact_keys(
            spec,
            {
                "provider_id", "instrument", "sequence_required", "first_sequence",
                "last_sequence", "event_count", "max_source_gap_ns", "max_arrival_gap_ns",
            },
            "stream policy",
        )
        provider = _text(spec["provider_id"], "stream provider")
        instrument = _text(spec["instrument"], "stream instrument")
        _pair(instrument)
        if provider not in providers or instrument not in registry:
            raise ReferenceError("stream outside allowlist/registry")
        if _boolean(spec["sequence_required"], "stream sequence required") is not True:
            raise ReferenceError("lossless stream must require sequence")
        for field in (
            "first_sequence", "last_sequence", "event_count",
            "max_source_gap_ns", "max_arrival_gap_ns",
        ):
            _integer(spec[field], f"stream.{field}", 1)
        key = (provider, instrument)
        if key in stream_policies:
            raise ReferenceError("duplicate stream policy")
        stream_policies[key] = dict(spec)
        policy_order.append(key)
    if policy_order != sorted(policy_order):
        raise ReferenceError("stream policies must be sorted")

    raw_lines = blob.splitlines(keepends=True)
    if len(raw_lines) > MAX_SOURCE_ROWS:
        raise ReferenceError("source row limit exceeded")
    expected_event_keys = {
        "schema_version", "provider_id", "instrument", "bid_ticks", "ask_ticks",
        "tick_scale", "source_ts_ns", "arrival_ts_ns", "provider_event_id", "sequence",
        "heartbeat", "quality_flags",
    }
    ticks: list[MarketTick] = []
    provider_event_ids: set[tuple[str, str, str]] = set()
    prior_global: tuple[int, int, str, str, int] | None = None
    prior_stream: dict[tuple[str, str], tuple[int, int, int]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    prefix = ZERO_SHA256
    for raw_line in raw_lines:
        row = _decode_json(raw_line, "source event")
        _exact_keys(row, expected_event_keys, "source event")
        if _integer(row["schema_version"], "source schema") != 1:
            raise ReferenceError("source event schema mismatch")
        provider = _text(row["provider_id"], "source provider")
        instrument = _text(row["instrument"], "source instrument")
        _pair(instrument)
        key = (provider, instrument)
        if key not in stream_policies:
            raise ReferenceError("source stream not frozen")
        for field in (
            "bid_ticks", "ask_ticks", "tick_scale", "source_ts_ns", "arrival_ts_ns", "sequence",
        ):
            _integer(row[field], f"source.{field}", 1)
        if row["ask_ticks"] <= row["bid_ticks"] \
                or row["arrival_ts_ns"] < row["source_ts_ns"]:
            raise ReferenceError("source BBO/clock invalid")
        if row["tick_scale"] != registry[instrument]["price_scale"]:
            raise ReferenceError("source price scale differs from registry")
        if row["provider_event_id"] is not None and type(row["provider_event_id"]) is not str:
            raise ReferenceError("provider event id type invalid")
        if row["provider_event_id"] is not None:
            provider_event_identity = (provider, instrument, row["provider_event_id"])
            if provider_event_identity in provider_event_ids:
                raise ReferenceError("duplicate provider event identity")
            provider_event_ids.add(provider_event_identity)
        if _boolean(row["heartbeat"], "source heartbeat") is not False \
                or type(row["quality_flags"]) is not list or row["quality_flags"]:
            raise ReferenceError("non-executable source event forbidden")
        order = (
            row["arrival_ts_ns"], row["source_ts_ns"], provider, instrument, row["sequence"],
        )
        if prior_global is not None and order <= prior_global:
            raise ReferenceError("global source order must be strictly increasing")
        prior_global = order
        prior = prior_stream.get(key)
        if prior is not None:
            policy = stream_policies[key]
            if row["source_ts_ns"] <= prior[0] \
                    or row["arrival_ts_ns"] <= prior[1] \
                    or row["sequence"] != prior[2] + 1 \
                    or row["source_ts_ns"] - prior[0] > policy["max_source_gap_ns"] \
                    or row["arrival_ts_ns"] - prior[1] > policy["max_arrival_gap_ns"]:
                raise ReferenceError("stream chronology/sequence/gap invalid")
        prior_stream[key] = (
            row["source_ts_ns"], row["arrival_ts_ns"], row["sequence"],
        )
        counts[key] += 1
        event_hash = sha256_bytes(raw_line)
        prefix = sha256_bytes(canonical_bytes({
            "previous_hash": prefix,
            "source_event_sha256": event_hash,
        }))
        ticks.append(MarketTick(
            provider_id=provider,
            instrument=instrument,
            bid_ticks=row["bid_ticks"],
            ask_ticks=row["ask_ticks"],
            tick_scale=row["tick_scale"],
            source_ts_ns=row["source_ts_ns"],
            arrival_ts_ns=row["arrival_ts_ns"],
            sequence=row["sequence"],
            source_event_sha256=event_hash,
            source_prefix_root_sha256=prefix,
        ))
    if _integer(manifest["event_count"], "source event count", 1) != len(ticks) \
            or _integer(manifest["first_source_ts_ns"], "first source time", 1) != min(t.source_ts_ns for t in ticks) \
            or _integer(manifest["last_source_ts_ns"], "last source time", 1) != max(t.source_ts_ns for t in ticks):
        raise ReferenceError("source manifest count/time mismatch")
    if set(stream_policies) != set(counts):
        raise ReferenceError("stream policy inventory mismatch")
    providers_by_instrument: defaultdict[str, set[str]] = defaultdict(set)
    for provider, instrument in stream_policies:
        providers_by_instrument[instrument].add(provider)
    if any(len(items) != 1 for items in providers_by_instrument.values()):
        raise ReferenceError("multiple providers per instrument forbidden")
    for key, spec in stream_policies.items():
        stream = [tick for tick in ticks if (tick.provider_id, tick.instrument) == key]
        if spec["first_sequence"] != stream[0].sequence \
                or spec["last_sequence"] != stream[-1].sequence \
                or spec["event_count"] != counts[key]:
            raise ReferenceError("stream policy count/sequence mismatch")
    books: defaultdict[str, list[MarketTick]] = defaultdict(list)
    for tick in ticks:
        books[tick.instrument].append(tick)
    return tuple(ticks), {key: tuple(value) for key, value in books.items()}


def _decode_proposals(
    raw: bytes,
    ticks: Sequence[MarketTick],
) -> tuple[dict[str, Any], tuple[Proposal, ...]]:
    payload = _decode_json(raw, "proposal")
    _reject_producer_fields(payload)
    _exact_keys(
        payload,
        {"schema_version", "candidate_key", "provenance", "rows", "proposal_sha256"},
        "proposal",
    )
    if _integer(payload["schema_version"], "proposal schema") != 2:
        raise ReferenceError("proposal schema mismatch")
    _require_embedded(payload, "proposal_sha256", "proposal")
    _text(payload["candidate_key"], "candidate key")
    provenance = payload["provenance"]
    _exact_keys(
        provenance,
        {
            "detector_code_sha256", "detector_policy_sha256",
            "generator_policy_sha256", "source_acquisition_contract_sha256",
        },
        "proposal provenance",
    )
    for key, digest in provenance.items():
        _digest(digest, f"proposal provenance {key}")
    rows = payload["rows"]
    if type(rows) is not list or not rows:
        raise ReferenceError("proposal rows missing")
    row_keys = {
        "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns", "available_at_ns",
        "decision_source_event_sha256", "completed_data_watermark_source_ts_ns",
        "completed_data_prefix_root_sha256", "instrument", "direction", "notional_jpy_micros",
        "max_age_ns", "worker_key", "action",
    }
    by_hash = {tick.source_event_sha256: tick for tick in ticks}
    proposals: list[Proposal] = []
    prior_order: tuple[int, int, int] | None = None
    economic_keys: set[str] = set()
    for expected_ordinal, row in enumerate(rows, 1):
        _exact_keys(row, row_keys, "proposal row")
        for field in (
            "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns", "available_at_ns",
            "completed_data_watermark_source_ts_ns", "direction", "notional_jpy_micros", "max_age_ns",
        ):
            _integer(row[field], f"proposal.{field}")
        if row["proposal_ordinal"] != expected_ordinal \
                or row["direction"] not in {-1, 1} \
                or row["notional_jpy_micros"] <= 0 \
                or row["max_age_ns"] <= 0 \
                or row["available_at_ns"] != row["decision_arrival_ts_ns"] \
                or row["decision_arrival_ts_ns"] < row["decision_source_ts_ns"] \
                or row["action"] != "ENTER":
            raise ReferenceError("proposal chronology/value invalid")
        instrument = _text(row["instrument"], "proposal instrument")
        _pair(instrument)
        worker_key = _text(row["worker_key"], "proposal worker")
        decision_hash = _digest(row["decision_source_event_sha256"], "decision event hash")
        prefix_hash = _digest(row["completed_data_prefix_root_sha256"], "proposal prefix hash")
        available = [tick for tick in ticks if tick.arrival_ts_ns <= row["decision_arrival_ts_ns"]]
        if not available:
            raise ReferenceError("proposal has no causal prefix")
        watermark = max(tick.source_ts_ns for tick in available)
        decision_tick = by_hash.get(decision_hash)
        if row["completed_data_watermark_source_ts_ns"] != watermark \
                or prefix_hash != available[-1].source_prefix_root_sha256 \
                or decision_tick is None or decision_tick not in available \
                or decision_tick.source_ts_ns != row["decision_source_ts_ns"] \
                or decision_tick.instrument != instrument:
            raise ReferenceError("proposal causal binding mismatch")
        order = (
            row["decision_arrival_ts_ns"], row["decision_source_ts_ns"], expected_ordinal,
        )
        if prior_order is not None and order <= prior_order:
            raise ReferenceError("proposal order reversal")
        prior_order = order
        economic_key = sha256_bytes(canonical_bytes({
            key: row[key] for key in sorted(row_keys - {"proposal_ordinal"})
        }))
        if economic_key in economic_keys:
            raise ReferenceError("duplicate economic-lot partition forbidden")
        economic_keys.add(economic_key)
        proposals.append(Proposal(
            ordinal=expected_ordinal,
            decision_source_ts_ns=row["decision_source_ts_ns"],
            decision_arrival_ts_ns=row["decision_arrival_ts_ns"],
            decision_source_event_sha256=decision_hash,
            completed_data_watermark_source_ts_ns=row[
                "completed_data_watermark_source_ts_ns"
            ],
            completed_data_prefix_root_sha256=prefix_hash,
            instrument=instrument,
            direction=row["direction"],
            target_notional_jpy_micros=row["notional_jpy_micros"],
            max_age_ns=row["max_age_ns"],
            worker_key=worker_key,
        ))
    return payload, tuple(proposals)


def _validate_policy_identity(
    payload: Mapping[str, Any], policy_id: str, hash_field: str,
) -> None:
    if _integer(payload.get("schema_version"), f"{policy_id} schema") != 2 \
            or payload.get("policy_id") != policy_id:
        raise ReferenceError(f"{policy_id} identity mismatch")
    _require_embedded(payload, hash_field, policy_id)


def _decode_policies(artifacts: Mapping[str, bytes]) -> tuple[
    dict[str, ArmTerms], int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str,
]:
    execution = _decode_json(artifacts["execution_policy"], "execution policy")
    inventory = _decode_json(artifacts["inventory_policy"], "inventory policy")
    accounting = _decode_json(artifacts["accounting_policy"], "accounting policy")
    evaluation = _decode_json(artifacts["evaluation_policy"], "evaluation policy")
    authority = _decode_json(artifacts["authority_policy"], "authority policy")

    _validate_policy_identity(execution, "FROZEN_EXECUTION_POLICY_V2", "execution_policy_sha256")
    _exact_keys(
        execution,
        {"schema_version", "policy_id", "arms", "max_trade_quote_staleness_ns", "execution_policy_sha256"},
        "execution policy",
    )
    max_stale = _integer(execution["max_trade_quote_staleness_ns"], "trade quote staleness", 1)
    if type(execution["arms"]) is not dict or set(execution["arms"]) != set(ARMS):
        raise ReferenceError("execution arm set mismatch")
    arms: dict[str, ArmTerms] = {}
    cost_fields = (
        "latency_ns", "slippage_micropips_per_side",
        "commission_ppm_per_side", "financing_ppm_per_day",
    )
    for arm in ARMS:
        spec = execution["arms"][arm]
        _exact_keys(spec, {*cost_fields, "raw_mid"}, f"execution arm {arm}")
        values = {field: _integer(spec[field], f"{arm}.{field}", 0) for field in cost_fields}
        raw_mid = _boolean(spec["raw_mid"], f"{arm}.raw_mid")
        arms[arm] = ArmTerms(**values, raw_mid=raw_mid)
    raw_arm, base_arm, adverse_arm = arms["RAW_SIGNAL"], arms["EXECUTABLE_BASE"], arms["ADVERSE_STRESS"]
    if raw_arm.raw_mid is not True or any(getattr(raw_arm, field) != 0 for field in cost_fields):
        raise ReferenceError("RAW arm must be zero-cost midpoint")
    if base_arm.raw_mid is not False or adverse_arm.raw_mid is not False \
            or any(getattr(adverse_arm, field) < getattr(base_arm, field) for field in cost_fields) \
            or not any(getattr(adverse_arm, field) > getattr(base_arm, field) for field in cost_fields):
        raise ReferenceError("adverse arm must be strictly harder")

    _validate_policy_identity(inventory, "FROZEN_INVENTORY_POLICY_V2", "inventory_policy_sha256")
    _exact_keys(
        inventory,
        {
            "schema_version", "policy_id", "max_gross_notional_jpy_micros",
            "max_currency_notional_jpy_micros", "max_open_positions", "same_pair_collision",
            "terminal_liquidation", "inventory_policy_sha256",
        },
        "inventory policy",
    )
    for field in (
        "max_gross_notional_jpy_micros", "max_currency_notional_jpy_micros", "max_open_positions",
    ):
        _integer(inventory[field], f"inventory.{field}", 1)
    if inventory["same_pair_collision"] != "REJECT_NEW" \
            or _boolean(inventory["terminal_liquidation"], "terminal liquidation") is not True:
        raise ReferenceError("inventory policy invalid")

    _validate_policy_identity(accounting, "FROZEN_ACCOUNTING_POLICY_V2", "accounting_policy_sha256")
    _exact_keys(
        accounting,
        {
            "schema_version", "policy_id", "jpy_micros_per_yen", "base_microunits_per_unit",
            "max_conversion_staleness_ns", "supported_quote_currencies", "asset_conversion_side",
            "liability_conversion_side", "positive_cost_rounding", "accounting_policy_sha256",
        },
        "accounting policy",
    )
    if _integer(accounting["jpy_micros_per_yen"], "JPY micros") != JPY_MICROS_PER_YEN \
            or _integer(accounting["base_microunits_per_unit"], "base microunits") != BASE_MICROUNITS_PER_UNIT \
            or _integer(accounting["max_conversion_staleness_ns"], "conversion staleness", 1) <= 0 \
            or accounting["supported_quote_currencies"] != ["CAD", "CHF", "JPY", "USD"] \
            or accounting["asset_conversion_side"] != "BID" \
            or accounting["liability_conversion_side"] != "ASK" \
            or accounting["positive_cost_rounding"] != "CEILING":
        raise ReferenceError("accounting policy invalid")

    _validate_policy_identity(evaluation, "FROZEN_EVALUATION_POLICY_V2", "evaluation_policy_sha256")
    _exact_keys(
        evaluation,
        {
            "schema_version", "policy_id", "period_start_ts_ns", "period_end_ts_ns",
            "initial_equity_jpy_micros", "margin_notional_cap_jpy_micros", "margin_rate_bps",
            "max_gross_to_equity_bps", "cvar_tail_bps", "cluster_window_ns", "full_month_ids",
            "holdout_state", "evaluation_policy_sha256",
        },
        "evaluation policy",
    )
    start = _integer(evaluation["period_start_ts_ns"], "period start", 1)
    end = _integer(evaluation["period_end_ts_ns"], "period end", 1)
    if start >= end:
        raise ReferenceError("evaluation period invalid")
    for field in (
        "initial_equity_jpy_micros", "margin_notional_cap_jpy_micros", "margin_rate_bps",
        "max_gross_to_equity_bps", "cvar_tail_bps", "cluster_window_ns",
    ):
        _integer(evaluation[field], f"evaluation.{field}", 1)
    if evaluation["margin_rate_bps"] > 10_000 \
            or evaluation["cvar_tail_bps"] > 10_000 \
            or evaluation["holdout_state"] != "UNOPENED":
        raise ReferenceError("evaluation risk/holdout policy invalid")
    complete_months = [
        month for month in _intersecting_months(start, end)
        if _month_bounds_ns(month)[0] >= start and _month_bounds_ns(month)[1] <= end
    ]
    if type(evaluation["full_month_ids"]) is not list \
            or evaluation["full_month_ids"] != complete_months:
        raise ReferenceError("full month set mismatch")

    _validate_policy_identity(authority, "FROZEN_PAPER_AUTHORITY_V1", "authority_policy_sha256")
    _exact_keys(
        authority,
        {
            "schema_version",
            "policy_id",
            "paper_only",
            "live_authority",
            "broker_account_access",
            "credential_access",
            "order_endpoint",
            "external_orders",
            "deploy",
            "external_config_mutation",
            "authority_policy_sha256",
        },
        "authority policy",
    )
    # These literal branches intentionally do not consult module state.  A
    # caller that can mutate this module's globals must still be unable to
    # redefine paper-only authority into live or broker authority.
    if type(authority["paper_only"]) is not bool \
            or authority["paper_only"] is not True:
        raise ReferenceError("authority.paper_only exact boolean mismatch")
    for key in (
        "live_authority",
        "broker_account_access",
        "credential_access",
        "order_endpoint",
        "deploy",
        "external_config_mutation",
    ):
        if type(authority[key]) is not bool or authority[key] is not False:
            raise ReferenceError(f"authority.{key} exact boolean mismatch")
    if type(authority["external_orders"]) is not int \
            or authority["external_orders"] != 0:
        raise ReferenceError("authority.external_orders exact integer mismatch")
    return (
        arms,
        max_stale,
        dict(inventory),
        dict(accounting),
        dict(evaluation),
        dict(authority),
        execution["execution_policy_sha256"],
    )


def decode_reference_input(artifacts: Mapping[str, bytes]) -> ReferenceInput:
    """Decode the exact nine-artifact economic boundary independently."""
    if not isinstance(artifacts, Mapping):
        raise ReferenceError("reference artifacts must be a mapping")
    items = tuple(artifacts.items())
    if (
        len(items) != len(ARTIFACT_KEYS)
        or any(type(label) is not str for label, _ in items)
        or {label for label, _ in items} != set(ARTIFACT_KEYS)
    ):
        raise ReferenceError("reference artifact set must be exactly the frozen nine")
    for label, raw in items:
        if type(raw) is not bytes:
            raise ReferenceError("reference artifacts must map text labels to exact bytes")
        ceiling = MAX_SOURCE_BYTES if label == "source_blob" else MAX_JSON_BYTES
        if not raw or len(raw) > ceiling:
            raise ReferenceError(f"{label} artifact size invalid")
    # Snapshot a general Mapping exactly once.  Subsequent parsing and hashing
    # consume only this immutable-by-convention byte snapshot, so a stateful
    # Mapping cannot make validation observe different bytes than replay.
    snapshot = dict(items)
    registry_payload, registry = _decode_registry(snapshot["instrument_registry"])
    ticks, books = _decode_source(
        snapshot["source_blob"],
        snapshot["source_manifest"],
        registry_payload,
        registry,
    )
    proposal_payload, proposals = _decode_proposals(snapshot["proposal"], ticks)
    arms, max_stale, inventory, accounting, evaluation, authority, execution_hash = (
        _decode_policies(snapshot)
    )
    if any(proposal.instrument not in registry for proposal in proposals):
        raise ReferenceError("proposal instrument outside registry")
    if any(
        proposal.decision_arrival_ts_ns < evaluation["period_start_ts_ns"]
        or proposal.decision_arrival_ts_ns >= evaluation["period_end_ts_ns"]
        for proposal in proposals
    ):
        raise ReferenceError("proposal outside evaluation period")
    return ReferenceInput(
        ticks=ticks,
        books=_deep_freeze(books),
        proposals=proposals,
        candidate_key=proposal_payload["candidate_key"],
        provenance=_deep_freeze(dict(proposal_payload["provenance"])),
        arms=_deep_freeze(arms),
        max_trade_quote_staleness_ns=max_stale,
        inventory=_deep_freeze(inventory),
        accounting=_deep_freeze(accounting),
        evaluation=_deep_freeze(evaluation),
        authority=_deep_freeze(authority),
        registry=_deep_freeze(registry),
        execution_policy_sha256=execution_hash,
        raw_hashes=_deep_freeze({
            key: sha256_bytes(snapshot[key]) for key in sorted(ARTIFACT_KEYS)
        }),
    )


def _tick_price(tick: MarketTick, side: str) -> Fraction:
    if side == "BID":
        return Fraction(tick.bid_ticks, tick.tick_scale)
    if side == "ASK":
        return Fraction(tick.ask_ticks, tick.tick_scale)
    if side == "MID":
        return Fraction(tick.bid_ticks + tick.ask_ticks, 2 * tick.tick_scale)
    raise ReferenceError("unknown quote side")


def _arrival_watermark(data: ReferenceInput, arrival_ns: int) -> int:
    values = [tick.source_ts_ns for tick in data.ticks if tick.arrival_ts_ns <= arrival_ns]
    if not values:
        raise ReferenceError("no causal source watermark")
    return max(values)


def _latest_fresh_tick(
    data: ReferenceInput,
    instrument: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    maximum_staleness_ns: int,
) -> MarketTick:
    eligible = [
        tick for tick in data.books.get(instrument, ())
        if tick.source_ts_ns <= source_watermark_ns
        and tick.arrival_ts_ns <= arrival_cutoff_ns
    ]
    if not eligible:
        raise ReferenceError(f"causal quote missing for {instrument}")
    tick = eligible[-1]
    if source_watermark_ns - tick.source_ts_ns > maximum_staleness_ns \
            or arrival_cutoff_ns - tick.arrival_ts_ns > maximum_staleness_ns \
            or arrival_cutoff_ns - tick.source_ts_ns > maximum_staleness_ns:
        raise ReferenceError(f"causal quote stale for {instrument}")
    return tick


def _conversion_paths(
    data: ReferenceInput,
    start_currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
) -> list[tuple[tuple[str, str, MarketTick], ...]]:
    adjacency: defaultdict[str, list[tuple[str, str, MarketTick]]] = defaultdict(list)
    for instrument in sorted(data.registry):
        if instrument not in data.books:
            continue
        try:
            tick = _latest_fresh_tick(
                data,
                instrument,
                source_watermark_ns,
                arrival_cutoff_ns,
                data.accounting["max_conversion_staleness_ns"],
            )
        except ReferenceError:
            continue
        base, quote = _pair(instrument)
        adjacency[base].append((quote, "BASE_TO_QUOTE", tick))
        adjacency[quote].append((base, "QUOTE_TO_BASE", tick))
    paths: list[tuple[tuple[str, str, MarketTick], ...]] = []

    def visit(
        currency: str,
        visited: frozenset[str],
        edges: tuple[tuple[str, str, MarketTick], ...],
    ) -> None:
        if currency == "JPY":
            paths.append(edges)
            return
        for destination, orientation, tick in sorted(
            adjacency.get(currency, ()),
            key=lambda edge: (edge[0], edge[1], edge[2].instrument),
        ):
            if destination in visited:
                continue
            visit(
                destination,
                visited | {destination},
                edges + ((destination, orientation, tick),),
            )

    visit(start_currency, frozenset({start_currency}), ())
    return paths


def _currency_node_yen(
    data: ReferenceInput,
    amount: Fraction,
    currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
) -> Fraction:
    if amount == 0 or currency == "JPY":
        return amount
    paths = _conversion_paths(data, currency, source_watermark_ns, arrival_cutoff_ns)
    if len(paths) != 1:
        raise ReferenceError("JPY conversion path must be uniquely causal")
    value = amount
    for _, orientation, tick in paths[0]:
        if orientation == "BASE_TO_QUOTE":
            value *= _tick_price(tick, "BID" if value > 0 else "ASK")
        else:
            value /= _tick_price(tick, "ASK" if value > 0 else "BID")
    return value


def _to_jpy_yen(
    data: ReferenceInput,
    amount: Fraction,
    currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
) -> Fraction:
    if amount != 0 and currency != "JPY" \
            and currency not in data.accounting["supported_quote_currencies"]:
        raise ReferenceError(f"unsupported conversion currency: {currency}")
    return _currency_node_yen(
        data,
        amount,
        currency,
        source_watermark_ns,
        arrival_cutoff_ns,
    )


def _execution_price(
    tick: MarketTick,
    proposal: Proposal,
    terms: ArmTerms,
    registry: Mapping[str, Mapping[str, int]],
    *,
    opening: bool,
) -> tuple[Fraction, int, int]:
    if terms.raw_mid:
        numerator = (tick.bid_ticks + tick.ask_ticks) * PRICE_SUBPIP_SCALE
        denominator = 2 * tick.tick_scale * PRICE_SUBPIP_SCALE
    else:
        buys_base = (opening and proposal.direction > 0) \
            or (not opening and proposal.direction < 0)
        market_ticks = tick.ask_ticks if buys_base else tick.bid_ticks
        slippage_ticks = (
            terms.slippage_micropips_per_side
            * registry[proposal.instrument]["pip_ticks"]
        )
        numerator = market_ticks * PRICE_SUBPIP_SCALE
        numerator += slippage_ticks if buys_base else -slippage_ticks
        denominator = tick.tick_scale * PRICE_SUBPIP_SCALE
    if numerator <= 0:
        raise ReferenceError("execution price nonpositive")
    return Fraction(numerator, denominator), numerator, denominator


def _is_trade_fresh(tick: MarketTick, due_ns: int, staleness_ns: int) -> bool:
    return tick.arrival_ts_ns >= due_ns \
        and tick.arrival_ts_ns - tick.source_ts_ns <= staleness_ns \
        and tick.arrival_ts_ns - due_ns <= staleness_ns


def _first_fill_tick(
    data: ReferenceInput,
    proposal: Proposal,
    latency_ns: int,
) -> MarketTick | None:
    due_ns = proposal.decision_arrival_ts_ns + latency_ns
    for tick in data.books.get(proposal.instrument, ()):
        if tick.source_ts_ns <= proposal.decision_source_ts_ns or tick.arrival_ts_ns < due_ns:
            continue
        if tick.arrival_ts_ns >= data.evaluation["period_end_ts_ns"]:
            return None
        if not _is_trade_fresh(tick, due_ns, data.max_trade_quote_staleness_ns):
            raise ReferenceError("first causal fill tick stale")
        return tick
    return None


def _first_close_tick(
    data: ReferenceInput,
    instrument: str,
    entry: MarketTick,
    due_ns: int,
) -> MarketTick | None:
    for tick in data.books.get(instrument, ()):
        if tick.source_ts_ns < entry.source_ts_ns or tick.arrival_ts_ns < due_ns:
            continue
        if tick.arrival_ts_ns >= data.evaluation["period_end_ts_ns"]:
            return None
        if not _is_trade_fresh(tick, due_ns, data.max_trade_quote_staleness_ns):
            raise ReferenceError("first causal close tick stale")
        return tick
    return None


def _terminal_tick(data: ReferenceInput, instrument: str) -> MarketTick:
    period_end = data.evaluation["period_end_ts_ns"]
    eligible = [
        tick for tick in data.books.get(instrument, ())
        if tick.source_ts_ns < period_end and tick.arrival_ts_ns < period_end
    ]
    if not eligible:
        raise ReferenceError("terminal tick missing")
    tick = eligible[-1]
    cutoff = period_end - 1
    if cutoff - tick.source_ts_ns > data.max_trade_quote_staleness_ns \
            or cutoff - tick.arrival_ts_ns > data.max_trade_quote_staleness_ns:
        raise ReferenceError("terminal tick stale")
    return tick


def _signal_id(data: ReferenceInput, proposal: Proposal) -> str:
    return sha256_bytes(canonical_bytes({
        "candidate_key": data.candidate_key,
        "proposal_ordinal": proposal.ordinal,
        "decision_source_ts_ns": proposal.decision_source_ts_ns,
        "decision_arrival_ts_ns": proposal.decision_arrival_ts_ns,
        "decision_source_event_sha256": proposal.decision_source_event_sha256,
        "completed_data_prefix_root_sha256": proposal.completed_data_prefix_root_sha256,
        "instrument": proposal.instrument,
        "direction": proposal.direction,
        "notional_jpy_micros": proposal.target_notional_jpy_micros,
        "max_age_ns": proposal.max_age_ns,
        "worker_key": proposal.worker_key,
        "detector_code_sha256": data.provenance["detector_code_sha256"],
        "detector_policy_sha256": data.provenance["detector_policy_sha256"],
        "generator_policy_sha256": data.provenance["generator_policy_sha256"],
    }))


def _economic_lot_id(data: ReferenceInput, proposal: Proposal) -> str:
    return sha256_bytes(canonical_bytes({
        "candidate_key": data.candidate_key,
        "decision_source_ts_ns": proposal.decision_source_ts_ns,
        "decision_arrival_ts_ns": proposal.decision_arrival_ts_ns,
        "decision_source_event_sha256": proposal.decision_source_event_sha256,
        "completed_data_prefix_root_sha256": proposal.completed_data_prefix_root_sha256,
        "instrument": proposal.instrument,
        "direction": proposal.direction,
        "target_notional_jpy_micros": proposal.target_notional_jpy_micros,
        "max_age_ns": proposal.max_age_ns,
        "worker_key": proposal.worker_key,
        "detector_code_sha256": data.provenance["detector_code_sha256"],
        "detector_policy_sha256": data.provenance["detector_policy_sha256"],
        "generator_policy_sha256": data.provenance["generator_policy_sha256"],
    }))


def _position_notional_exact(
    data: ReferenceInput,
    proposal: Proposal,
    units_micros: int,
    price: Fraction,
    arrival_ns: int,
    *,
    opening: bool,
) -> Fraction:
    _, quote_currency = _pair(proposal.instrument)
    source_watermark = _arrival_watermark(data, arrival_ns)
    cash_direction = -proposal.direction if opening else proposal.direction
    quote_amount = Fraction(
        cash_direction * units_micros,
        BASE_MICROUNITS_PER_UNIT,
    ) * price
    return abs(
        _to_jpy_yen(
            data,
            quote_amount,
            quote_currency,
            source_watermark,
            arrival_ns,
        ) * JPY_MICROS_PER_YEN
    )


def _sized_units(
    data: ReferenceInput,
    proposal: Proposal,
    tick: MarketTick,
    price: Fraction,
) -> int:
    per_unit = _position_notional_exact(
        data,
        proposal,
        BASE_MICROUNITS_PER_UNIT,
        price,
        tick.arrival_ts_ns,
        opening=True,
    )
    if per_unit <= 0:
        raise ReferenceError("sizing conversion nonpositive")
    exact = Fraction(
        proposal.target_notional_jpy_micros * BASE_MICROUNITS_PER_UNIT,
        1,
    ) / per_unit
    return max(0, _floor_fraction(exact))


def _common_reference(data: ReferenceInput, proposal: Proposal) -> Mapping[str, Any] | None:
    entry = _first_fill_tick(data, proposal, 0)
    if entry is None:
        return None
    due_ns = entry.arrival_ts_ns + proposal.max_age_ns
    exit_tick = _first_close_tick(data, proposal.instrument, entry, due_ns)
    exit_reason = "FINITE_MAX_AGE"
    valuation_arrival_ns = exit_tick.arrival_ts_ns if exit_tick is not None else None
    if exit_tick is None:
        exit_tick = _terminal_tick(data, proposal.instrument)
        if exit_tick.arrival_ts_ns < entry.arrival_ts_ns:
            raise ReferenceError("terminal reference precedes entry")
        exit_reason = "TERMINAL_LIQUIDATION"
        valuation_arrival_ns = data.evaluation["period_end_ts_ns"] - 1
    entry_mid = _tick_price(entry, "MID")
    units = _sized_units(data, proposal, entry, entry_mid)
    gross = 0
    if units:
        _, quote_currency = _pair(proposal.instrument)
        quote_pnl = Fraction(
            proposal.direction * units,
            BASE_MICROUNITS_PER_UNIT,
        ) * (_tick_price(exit_tick, "MID") - entry_mid)
        source_watermark = _arrival_watermark(data, valuation_arrival_ns)
        gross = _floor_fraction(
            _to_jpy_yen(
                data,
                quote_pnl,
                quote_currency,
                source_watermark,
                valuation_arrival_ns,
            ) * JPY_MICROS_PER_YEN
        )
    return {
        "entry": entry,
        "exit": exit_tick,
        "exit_reason": exit_reason,
        "exit_valuation_arrival_ns": valuation_arrival_ns,
        "units_micros": units,
        "gross_pnl_jpy_micros": gross,
    }


class _Journal:
    def __init__(self) -> None:
        self.transactions: list[JournalTransaction] = []
        self.event_ids: set[str] = set()
        self.last_arrival_by_arm: dict[str, int] = {}
        self.committed_dispositions: dict[tuple[str, int], Disposition] = {}

    def post(
        self,
        *,
        arrival_ts_ns: int,
        arm: str,
        proposal_ordinal: int,
        event_kind: str,
        event_id: str,
        source_event_sha256: str | None,
        postings: Iterable[tuple[str, Fraction]],
    ) -> None:
        combined: defaultdict[str, Fraction] = defaultdict(Fraction)
        for account, amount in postings:
            if account not in JOURNAL_ACCOUNTS:
                raise ReferenceError("unknown journal account")
            combined[account] += Fraction(amount)
        normalized = tuple(
            Posting(account, combined[account])
            for account in JOURNAL_ACCOUNT_ORDER
            if combined[account]
        )
        # A semantically empty disposition has no accounting event.  Omitting
        # it makes replays canonical without manufacturing zero-value legs.
        if not normalized:
            return
        if event_id in self.event_ids:
            raise ReferenceError("duplicate journal event")
        previous_arrival = self.last_arrival_by_arm.get(arm)
        if previous_arrival is not None and arrival_ts_ns < previous_arrival:
            raise ReferenceError("journal arm-local clock reversal")
        if len(normalized) < 2 \
                or sum((posting.amount for posting in normalized), Fraction()) != 0:
            raise ReferenceError("journal transaction is not exactly balanced")
        self.event_ids.add(event_id)
        self.last_arrival_by_arm[arm] = arrival_ts_ns
        self.transactions.append(JournalTransaction(
            sequence=len(self.transactions) + 1,
            arrival_ts_ns=arrival_ts_ns,
            arm=arm,
            proposal_ordinal=proposal_ordinal,
            event_kind=event_kind,
            event_id=event_id,
            source_event_sha256=source_event_sha256,
            postings=normalized,
        ))

    def commit_disposition(
        self,
        disposition: Disposition,
        settlement_arrival_ns: int,
    ) -> None:
        arm = (
            disposition.position.arm
            if isinstance(disposition, ClosedDisposition)
            else disposition.arm
        )
        ordinal = disposition.position.proposal.ordinal if isinstance(
            disposition, ClosedDisposition
        ) else disposition.proposal.ordinal
        key = (arm, ordinal)
        if key in self.committed_dispositions:
            raise ReferenceError("duplicate committed disposition")
        if disposition.settlement_arrival_ns != settlement_arrival_ns:
            raise ReferenceError("disposition settlement clock mismatch")
        previous_arrival = self.last_arrival_by_arm.get(arm)
        if previous_arrival is not None and settlement_arrival_ns < previous_arrival:
            raise ReferenceError("disposition arm-local clock reversal")
        self.last_arrival_by_arm[arm] = settlement_arrival_ns
        self.committed_dispositions[key] = disposition

    def arm_dispositions(self, arm: str) -> tuple[Disposition, ...]:
        return tuple(
            self.committed_dispositions[(arm, ordinal)]
            for ordinal in sorted(
                ordinal
                for event_arm, ordinal in self.committed_dispositions
                if event_arm == arm
            )
        )

    def balances(self) -> dict[str, dict[str, Fraction]]:
        result: defaultdict[str, defaultdict[str, Fraction]] = defaultdict(
            lambda: defaultdict(Fraction)
        )
        for transaction in self.transactions:
            for posting in transaction.postings:
                result[transaction.arm][posting.account] += posting.amount
        return {
            arm: dict(accounts)
            for arm, accounts in sorted(result.items())
        }

    def root(self) -> str:
        previous = ZERO_SHA256
        for transaction in self.transactions:
            payload = {
                "journal_schema_version": 1,
                "sequence": transaction.sequence,
                "previous_hash": previous,
                "arrival_ts_ns": transaction.arrival_ts_ns,
                "arm": transaction.arm,
                "proposal_ordinal": transaction.proposal_ordinal,
                "event_kind": transaction.event_kind,
                "event_id": transaction.event_id,
                "source_event_sha256": transaction.source_event_sha256,
                "postings": [
                    {
                        "account": posting.account,
                        "amount_numerator": posting.amount.numerator,
                        "amount_denominator": posting.amount.denominator,
                    }
                    for posting in transaction.postings
                ],
            }
            previous = sha256_bytes(canonical_bytes(payload))
        return previous


def _common_gross_for_units(
    data: ReferenceInput,
    proposal: Proposal,
    common: Mapping[str, Any],
    units_micros: int,
) -> int:
    if units_micros < 0:
        raise ReferenceError("common-path unit count negative")
    if units_micros == 0:
        return 0
    entry: MarketTick = common["entry"]
    exit_tick: MarketTick = common["exit"]
    valuation_arrival_ns = common["exit_valuation_arrival_ns"]
    _, quote_currency = _pair(proposal.instrument)
    quote_pnl = Fraction(
        proposal.direction * units_micros,
        BASE_MICROUNITS_PER_UNIT,
    ) * (_tick_price(exit_tick, "MID") - _tick_price(entry, "MID"))
    return _floor_fraction(
        _to_jpy_yen(
            data,
            quote_pnl,
            quote_currency,
            _arrival_watermark(data, valuation_arrival_ns),
            valuation_arrival_ns,
        ) * JPY_MICROS_PER_YEN
    )


def _position_values(
    data: ReferenceInput,
    position: PositionLot,
    mark_tick: MarketTick,
    *,
    valuation_arrival_ns: int | None = None,
    valuation_source_watermark_ns: int | None = None,
) -> dict[str, Any]:
    arrival_ns = mark_tick.arrival_ts_ns if valuation_arrival_ns is None else valuation_arrival_ns
    watermark = (
        _arrival_watermark(data, arrival_ns)
        if valuation_source_watermark_ns is None
        else valuation_source_watermark_ns
    )
    terms = data.arms[position.arm]
    exit_price, exit_num, exit_den = _execution_price(
        mark_tick,
        position.proposal,
        terms,
        data.registry,
        opening=False,
    )
    _, quote_currency = _pair(position.proposal.instrument)
    quote_pnl = Fraction(
        position.proposal.direction * position.units_micros,
        BASE_MICROUNITS_PER_UNIT,
    ) * (exit_price - position.entry_price)
    executable_exact = _to_jpy_yen(
        data,
        quote_pnl,
        quote_currency,
        watermark,
        arrival_ns,
    ) * JPY_MICROS_PER_YEN
    marked_notional_exact = _position_notional_exact(
        data,
        position.proposal,
        position.units_micros,
        exit_price,
        arrival_ns,
        opening=False,
    )
    elapsed_ns = arrival_ns - position.entry.arrival_ts_ns
    if elapsed_ns < 0:
        raise ReferenceError("valuation precedes entry")
    entry_commission_exact = position.entry_notional_exact \
        * terms.commission_ppm_per_side / 1_000_000
    exit_commission_exact = marked_notional_exact \
        * terms.commission_ppm_per_side / 1_000_000
    financing_exact = position.entry_notional_exact \
        * terms.financing_ppm_per_day * elapsed_ns / (DAY_NS * 1_000_000)
    entry_commission = _ceil_nonnegative(entry_commission_exact)
    exit_commission = _ceil_nonnegative(exit_commission_exact)
    financing = _ceil_nonnegative(financing_exact)
    executable = _floor_fraction(executable_exact)
    economic_net_exact = (
        executable_exact - entry_commission_exact - exit_commission_exact - financing_exact
    )
    return {
        "exit_price": exit_price,
        "exit_price_numerator": exit_num,
        "exit_price_denominator": exit_den,
        "executable_exact": executable_exact,
        "executable_pnl_jpy_micros": executable,
        "entry_commission_exact": entry_commission_exact,
        "exit_commission_exact": exit_commission_exact,
        "commission_jpy_micros": entry_commission + exit_commission,
        "financing_exact": financing_exact,
        "financing_jpy_micros": financing,
        "net_pnl_jpy_micros": executable - entry_commission - exit_commission - financing,
        "economic_net_exact": economic_net_exact,
        "elapsed_ns": elapsed_ns,
        "marked_notional_exact": marked_notional_exact,
        "marked_notional_jpy_micros": _ceil_nonnegative(marked_notional_exact),
        "financing_basis_notional_jpy_micros": _ceil_nonnegative(
            position.entry_notional_exact
        ),
    }


def _currency_exposure_postings(
    positions: Sequence[PositionLot],
    marked: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, Fraction], ...]:
    if len(positions) != len(marked):
        raise ReferenceError("position/mark count mismatch in currency book")
    postings: list[tuple[str, str, Fraction]] = []
    for position, mark in zip(positions, marked):
        base, quote = _pair(position.proposal.instrument)
        base_native = Fraction(
            position.proposal.direction * position.units_micros,
            BASE_MICROUNITS_PER_UNIT,
        )
        mark_price = mark.get("exit_price")
        if not isinstance(mark_price, Fraction) or mark_price <= 0:
            raise ReferenceError("currency posting mark price invalid")
        quote_native = -base_native * mark_price
        if base_native * position.proposal.direction <= 0 \
                or quote_native * position.proposal.direction >= 0:
            raise ReferenceError("currency posting sign invariant failed")
        postings.append(("BASE_POSITION", base, base_native))
        postings.append(("QUOTE_COUNTERVALUE", quote, quote_native))
    return tuple(postings)


def _signed_exposure(
    data: ReferenceInput,
    positions: Sequence[PositionLot],
    marked: Sequence[Mapping[str, Any]],
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
) -> dict[str, int]:
    values: defaultdict[str, Fraction] = defaultdict(Fraction)
    for posting_kind, currency, native_amount in _currency_exposure_postings(
        positions, marked
    ):
        if posting_kind not in {"BASE_POSITION", "QUOTE_COUNTERVALUE"}:
            raise ReferenceError("unknown currency exposure posting")
        if native_amount:
            values[currency] += _currency_node_yen(
                data,
                native_amount,
                currency,
                source_watermark_ns,
                arrival_cutoff_ns,
            ) * JPY_MICROS_PER_YEN
    rounded = {
        currency: (
            _ceil_nonnegative(exact_micros)
            if exact_micros >= 0
            else _floor_fraction(exact_micros)
        )
        for currency, exact_micros in sorted(values.items())
    }
    return {currency: amount for currency, amount in rounded.items() if amount}


def _mark_state(
    data: ReferenceInput,
    active: Sequence[PositionLot],
    closed_dispositions: Sequence[ClosedDisposition],
    arrival_ns: int,
) -> RiskSnapshot:
    watermark = _arrival_watermark(data, arrival_ns)
    marks: list[dict[str, Any]] = []
    for position in active:
        quote = _latest_fresh_tick(
            data,
            position.proposal.instrument,
            watermark,
            arrival_ns,
            data.max_trade_quote_staleness_ns,
        )
        marks.append(_position_values(
            data,
            position,
            quote,
            valuation_arrival_ns=arrival_ns,
            valuation_source_watermark_ns=watermark,
        ))
    realized = sum(
        disposition.values["net_pnl_jpy_micros"]
        for disposition in closed_dispositions
    )
    unrealized = sum(mark["net_pnl_jpy_micros"] for mark in marks)
    equity = data.evaluation["initial_equity_jpy_micros"] + realized + unrealized
    gross = sum(mark["marked_notional_jpy_micros"] for mark in marks)
    required = _ceil_nonnegative(Fraction(
        gross * data.evaluation["margin_rate_bps"],
        10_000,
    ))
    free = equity - required
    ratio_pass = equity > 0 \
        and gross * 10_000 <= equity * data.evaluation["max_gross_to_equity_bps"]
    return RiskSnapshot(
        arrival_ts_ns=arrival_ns,
        source_watermark_ts_ns=watermark,
        marked_equity_jpy_micros=equity,
        gross_notional_jpy_micros=gross,
        required_margin_jpy_micros=required,
        free_margin_jpy_micros=free,
        signed_currency_exposure_jpy_micros=MappingProxyType(
            _signed_exposure(data, active, marks, watermark, arrival_ns)
        ),
        margin_ratio_pass=ratio_pass,
    )


def _risk_closeout_reason(
    data: ReferenceInput, mark: RiskSnapshot
) -> str | None:
    if mark.marked_equity_jpy_micros <= 0 \
            or mark.free_margin_jpy_micros < 0 \
            or mark.margin_ratio_pass is not True \
            or mark.gross_notional_jpy_micros > data.evaluation[
                "margin_notional_cap_jpy_micros"
            ]:
        return "MARGIN_CLOSEOUT"
    currency_peak = max((
        abs(amount)
        for amount in mark.signed_currency_exposure_jpy_micros.values()
    ), default=0)
    if mark.gross_notional_jpy_micros \
            > data.inventory["max_gross_notional_jpy_micros"] \
            or currency_peak \
                > data.inventory["max_currency_notional_jpy_micros"]:
        return "INVENTORY_CAP_CLOSEOUT"
    return None


def _journal_entry(data: ReferenceInput, journal: _Journal, position: PositionLot) -> None:
    basis = position.entry_notional_exact
    journal.post(
        arrival_ts_ns=position.entry.arrival_ts_ns,
        arm=position.arm,
        proposal_ordinal=position.proposal.ordinal,
        event_kind="POSITION_OPEN",
        event_id=f"{position.arm}:{position.proposal.ordinal}:OPEN:{position.entry.source_event_sha256}",
        source_event_sha256=position.entry.source_event_sha256,
        postings=(("POSITION_BASIS", basis), ("POSITION_CONTROL", -basis)),
    )
    if position.entry_commission_exact:
        journal.post(
            arrival_ts_ns=position.entry.arrival_ts_ns,
            arm=position.arm,
            proposal_ordinal=position.proposal.ordinal,
            event_kind="ENTRY_COMMISSION",
            event_id=f"{position.arm}:{position.proposal.ordinal}:ENTRY_COMMISSION",
            source_event_sha256=position.entry.source_event_sha256,
            postings=(
                ("COMMISSION_EXPENSE", position.entry_commission_exact),
                ("SETTLEMENT_CASH", -position.entry_commission_exact),
            ),
        )


def _journal_mark(
    data: ReferenceInput,
    journal: _Journal,
    position: PositionLot,
    tick: MarketTick,
    arrival_ns: int,
) -> None:
    if position.last_mark_arrival_ns == arrival_ns:
        return
    values = _position_values(data, position, tick, valuation_arrival_ns=arrival_ns)
    delta = values["executable_exact"] - position.last_mark_pnl_exact
    position.last_mark_arrival_ns = arrival_ns
    if not delta:
        return
    journal.post(
        arrival_ts_ns=arrival_ns,
        arm=position.arm,
        proposal_ordinal=position.proposal.ordinal,
        event_kind="MARK_DELTA",
        event_id=f"{position.arm}:{position.proposal.ordinal}:MARK:{arrival_ns}",
        source_event_sha256=tick.source_event_sha256,
        postings=(("UNREALIZED_ASSET", delta), ("UNREALIZED_PNL", -delta)),
    )
    position.last_mark_pnl_exact = values["executable_exact"]


def _journal_close(
    journal: _Journal,
    position: PositionLot,
    exit_tick: MarketTick,
    arrival_ns: int,
    values: Mapping[str, Any],
    common_gross: int,
    fill_sizing_drag: int,
    execution_drag: int,
) -> None:
    suffix = f"{position.arm}:{position.proposal.ordinal}:CLOSE:{arrival_ns}"
    if position.last_mark_pnl_exact:
        journal.post(
            arrival_ts_ns=arrival_ns,
            arm=position.arm,
            proposal_ordinal=position.proposal.ordinal,
            event_kind="REVERSE_UNREALIZED",
            event_id=suffix + ":REVERSE_MARK",
            source_event_sha256=exit_tick.source_event_sha256,
            postings=(
                ("UNREALIZED_ASSET", -position.last_mark_pnl_exact),
                ("UNREALIZED_PNL", position.last_mark_pnl_exact),
            ),
        )
    journal.post(
        arrival_ts_ns=arrival_ns,
        arm=position.arm,
        proposal_ordinal=position.proposal.ordinal,
        event_kind="POSITION_CLOSE",
        event_id=suffix + ":BASIS",
        source_event_sha256=exit_tick.source_event_sha256,
        postings=(
            ("POSITION_BASIS", -position.entry_notional_exact),
            ("POSITION_CONTROL", position.entry_notional_exact),
        ),
    )
    executable_exact: Fraction = values["executable_exact"]
    if executable_exact:
        journal.post(
            arrival_ts_ns=arrival_ns,
            arm=position.arm,
            proposal_ordinal=position.proposal.ordinal,
            event_kind="REALIZE_TRADING_PNL",
            event_id=suffix + ":REALIZE",
            source_event_sha256=exit_tick.source_event_sha256,
            postings=(
                ("SETTLEMENT_CASH", executable_exact),
                ("REALIZED_TRADING_PNL", -executable_exact),
            ),
        )
    exit_commission: Fraction = values["exit_commission_exact"]
    if exit_commission:
        journal.post(
            arrival_ts_ns=arrival_ns,
            arm=position.arm,
            proposal_ordinal=position.proposal.ordinal,
            event_kind="EXIT_COMMISSION",
            event_id=suffix + ":EXIT_COMMISSION",
            source_event_sha256=exit_tick.source_event_sha256,
            postings=(
                ("COMMISSION_EXPENSE", exit_commission),
                ("SETTLEMENT_CASH", -exit_commission),
            ),
        )
    financing: Fraction = values["financing_exact"]
    if financing:
        journal.post(
            arrival_ts_ns=arrival_ns,
            arm=position.arm,
            proposal_ordinal=position.proposal.ordinal,
            event_kind="FINANCING",
            event_id=suffix + ":FINANCING",
            source_event_sha256=exit_tick.source_event_sha256,
            postings=(
                ("FINANCING_EXPENSE", financing),
                ("SETTLEMENT_CASH", -financing),
            ),
        )
    direct_cost = values["commission_jpy_micros"] + values["financing_jpy_micros"]
    net = values["net_pnl_jpy_micros"]
    journal.post(
        arrival_ts_ns=arrival_ns,
        arm=position.arm,
        proposal_ordinal=position.proposal.ordinal,
        event_kind="ATTRIBUTION_CONTROL",
        event_id=suffix + ":ATTRIBUTION",
        source_event_sha256=exit_tick.source_event_sha256,
        postings=(
            ("COMMON_GROSS_REFERENCE", Fraction(common_gross)),
            ("FILL_SIZING_DRAG", Fraction(-fill_sizing_drag)),
            ("LATENCY_SPREAD_SLIPPAGE_DRAG", Fraction(-execution_drag)),
            ("DIRECT_COST", Fraction(-direct_cost)),
            ("NET_PNL_CONTROL", Fraction(-net)),
        ),
    )


def _settle_position(
    data: ReferenceInput,
    journal: _Journal,
    position: PositionLot,
    exit_tick: MarketTick,
    exit_reason: str,
    *,
    valuation_arrival_ns: int | None = None,
    valuation_source_watermark_ns: int | None = None,
) -> ClosedDisposition:
    mutable_values = _position_values(
        data,
        position,
        exit_tick,
        valuation_arrival_ns=valuation_arrival_ns,
        valuation_source_watermark_ns=valuation_source_watermark_ns,
    )
    actual_arrival_ns = (
        exit_tick.arrival_ts_ns if valuation_arrival_ns is None else valuation_arrival_ns
    )
    common_gross = position.common["gross_pnl_jpy_micros"]
    arm_common_gross = _common_gross_for_units(
        data,
        position.proposal,
        position.common,
        position.units_micros,
    )
    fill_sizing_drag = common_gross - arm_common_gross
    execution_drag = arm_common_gross - mutable_values[
        "executable_pnl_jpy_micros"
    ]
    _journal_close(
        journal,
        position,
        exit_tick,
        actual_arrival_ns,
        mutable_values,
        common_gross,
        fill_sizing_drag,
        execution_drag,
    )
    disposition = ClosedDisposition(
        position=position,
        exit_tick=exit_tick,
        exit_reason=exit_reason,
        settlement_arrival_ns=actual_arrival_ns,
        values=MappingProxyType(dict(mutable_values)),
        common_gross_jpy_micros=common_gross,
        arm_common_gross_jpy_micros=arm_common_gross,
        fill_sizing_drag_jpy_micros=fill_sizing_drag,
        execution_drag_jpy_micros=execution_drag,
    )
    journal.commit_disposition(disposition, actual_arrival_ns)
    return disposition


def _project_closed_disposition(
    data: ReferenceInput,
    disposition: ClosedDisposition,
) -> dict[str, Any]:
    position = disposition.position
    exit_tick = disposition.exit_tick
    values = disposition.values
    actual_arrival_ns = disposition.settlement_arrival_ns
    common_gross = disposition.common_gross_jpy_micros
    arm_common_gross = disposition.arm_common_gross_jpy_micros
    fill_sizing_drag = disposition.fill_sizing_drag_jpy_micros
    execution_drag = disposition.execution_drag_jpy_micros
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": position.arm,
        "signal_id": position.signal_id,
        "proposal_ordinal": position.proposal.ordinal,
        "instrument": position.proposal.instrument,
        "direction": position.proposal.direction,
        "status": "FILLED_CLOSED",
        "entry_disposition": "FILLED",
        "exit_disposition": disposition.exit_reason,
        "action_transitions": ["ENTER", "EXIT"],
        "notional_jpy_micros": position.proposal.target_notional_jpy_micros,
        "target_notional_jpy_micros": position.proposal.target_notional_jpy_micros,
        "filled_notional_jpy_micros": position.entry_notional_rounded,
        "financing_basis_notional_jpy_micros": values[
            "financing_basis_notional_jpy_micros"
        ],
        "marked_or_exit_notional_jpy_micros": values["marked_notional_jpy_micros"],
        "exit_notional_jpy_micros": values["marked_notional_jpy_micros"],
        "units_micros": position.units_micros,
        "economic_lot_id": position.economic_lot_id,
        "common_entry_source_event_sha256": position.common["entry"].source_event_sha256,
        "common_exit_source_event_sha256": position.common["exit"].source_event_sha256,
        "common_gross_pnl_jpy_micros": common_gross,
        "arm_units_common_gross_pnl_jpy_micros": arm_common_gross,
        "entry_price_numerator": position.entry_price_numerator,
        "entry_price_denominator": position.entry_price_denominator,
        "exit_price_numerator": values["exit_price_numerator"],
        "exit_price_denominator": values["exit_price_denominator"],
        "entry_source_event_sha256": position.entry.source_event_sha256,
        "entry_source_ts_ns": position.entry.source_ts_ns,
        "entry_arrival_ts_ns": position.entry.arrival_ts_ns,
        "exit_source_event_sha256": exit_tick.source_event_sha256,
        "exit_source_ts_ns": exit_tick.source_ts_ns,
        "exit_arrival_ts_ns": actual_arrival_ns,
        "elapsed_ns": values["elapsed_ns"],
        "executable_pnl_before_direct_cost_jpy_micros": values[
            "executable_pnl_jpy_micros"
        ],
        "fill_sizing_drag_jpy_micros": fill_sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": execution_drag,
        "commission_jpy_micros": values["commission_jpy_micros"],
        "financing_jpy_micros": values["financing_jpy_micros"],
        "realized_cost_jpy_micros": common_gross - values["net_pnl_jpy_micros"],
        "admission_opportunity_drag_jpy_micros": 0,
        "net_pnl_jpy_micros": values["net_pnl_jpy_micros"],
        "economic_net_pnl_jpy_micros_numerator": values["economic_net_exact"].numerator,
        "economic_net_pnl_jpy_micros_denominator": values["economic_net_exact"].denominator,
        "signed_currency_exposure_after_entry_jpy_micros": dict(
            position.signed_exposure_after_entry or {}
        ),
        "gross_open_notional_after_entry_jpy_micros": position.gross_after_entry,
        "marked_equity_after_entry_jpy_micros": position.marked_equity_after_entry,
        "required_margin_after_entry_jpy_micros": position.required_margin_after_entry,
        "free_margin_after_entry_jpy_micros": position.free_margin_after_entry,
        "entry_source_reference": {
            "provider_id": position.entry.provider_id,
            "source_event_sha256": position.entry.source_event_sha256,
            "source_ts_ns": position.entry.source_ts_ns,
            "arrival_ts_ns": position.entry.arrival_ts_ns,
            "execution_policy_sha256": data.execution_policy_sha256,
        },
        "exit_source_reference": {
            "provider_id": exit_tick.provider_id,
            "source_event_sha256": exit_tick.source_event_sha256,
            "source_ts_ns": exit_tick.source_ts_ns,
            "arrival_ts_ns": exit_tick.arrival_ts_ns,
            "execution_policy_sha256": data.execution_policy_sha256,
        },
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _rejection_amounts(
    disposition: RejectedDisposition,
) -> tuple[int, int, int, int, int]:
    gross = (
        0
        if disposition.common is None
        else disposition.common["gross_pnl_jpy_micros"]
    )
    reason = disposition.reason
    sizing_drag = gross if reason == "SIZE_ROUNDED_TO_ZERO" else 0
    latency_drag = gross if reason == "NO_CAUSAL_FILL" else 0
    admission_drag = gross if reason in {
        "SAME_PAIR_COLLISION_REJECTED", "GROSS_CAP_REJECTED", "POSITION_CAP_REJECTED",
        "CURRENCY_CAP_REJECTED", "MARGIN_ENTRY_REJECTED", "ACCOUNT_HALTED",
    } else 0
    arm_common = 0 if reason in {"NO_CAUSAL_FILL", "SIZE_ROUNDED_TO_ZERO"} else gross
    return gross, sizing_drag, latency_drag, admission_drag, arm_common


def _project_rejected_disposition(
    disposition: RejectedDisposition,
) -> dict[str, Any]:
    proposal = disposition.proposal
    common = disposition.common
    gross, sizing_drag, latency_drag, admission_drag, arm_common = (
        _rejection_amounts(disposition)
    )
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": disposition.arm,
        "signal_id": disposition.signal_id,
        "proposal_ordinal": proposal.ordinal,
        "instrument": proposal.instrument,
        "direction": proposal.direction,
        "status": disposition.reason,
        "entry_disposition": disposition.reason,
        "exit_disposition": "NOT_APPLICABLE",
        "action_transitions": ["NO_ENTRY"],
        "notional_jpy_micros": proposal.target_notional_jpy_micros,
        "target_notional_jpy_micros": proposal.target_notional_jpy_micros,
        "filled_notional_jpy_micros": 0,
        "financing_basis_notional_jpy_micros": 0,
        "marked_or_exit_notional_jpy_micros": 0,
        "exit_notional_jpy_micros": 0,
        "units_micros": 0,
        "economic_lot_id": disposition.economic_lot_id,
        "common_entry_source_event_sha256": None if common is None else common["entry"].source_event_sha256,
        "common_exit_source_event_sha256": None if common is None else common["exit"].source_event_sha256,
        "common_gross_pnl_jpy_micros": gross,
        "arm_units_common_gross_pnl_jpy_micros": arm_common,
        "executable_pnl_before_direct_cost_jpy_micros": 0,
        "fill_sizing_drag_jpy_micros": sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": latency_drag,
        "commission_jpy_micros": 0,
        "financing_jpy_micros": 0,
        "realized_cost_jpy_micros": 0,
        "admission_opportunity_drag_jpy_micros": admission_drag,
        "net_pnl_jpy_micros": 0,
        "economic_net_pnl_jpy_micros_numerator": 0,
        "economic_net_pnl_jpy_micros_denominator": 1,
        "signed_currency_exposure_after_entry_jpy_micros": {},
        "gross_open_notional_after_entry_jpy_micros": 0,
        "marked_equity_after_entry_jpy_micros": None,
        "required_margin_after_entry_jpy_micros": 0,
        "free_margin_after_entry_jpy_micros": None,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _settle_rejection(
    journal: _Journal,
    pending: PendingRejection,
) -> RejectedDisposition:
    disposition = RejectedDisposition(
        arm=pending.arm,
        proposal=pending.proposal,
        signal_id=pending.signal_id,
        economic_lot_id=pending.economic_lot_id,
        reason=pending.reason,
        common=pending.common,
        known_arrival_ns=pending.known_arrival_ns,
        settlement_arrival_ns=pending.settlement_arrival_ns,
    )
    gross, sizing_drag, latency_drag, admission_drag, _ = _rejection_amounts(
        disposition
    )
    journal.post(
        arrival_ts_ns=disposition.settlement_arrival_ns,
        arm=disposition.arm,
        proposal_ordinal=disposition.proposal.ordinal,
        event_kind="PROPOSAL_DISPOSITION",
        event_id=(
            f"{disposition.arm}:{disposition.proposal.ordinal}:REJECT:"
            f"{disposition.reason}"
        ),
        source_event_sha256=(
            None
            if disposition.common is None
            else disposition.common["entry"].source_event_sha256
        ),
        postings=(
            ("COMMON_GROSS_REFERENCE", Fraction(gross)),
            ("FILL_SIZING_DRAG", Fraction(-sizing_drag)),
            ("LATENCY_SPREAD_SLIPPAGE_DRAG", Fraction(-latency_drag)),
            ("ADMISSION_OPPORTUNITY_DRAG", Fraction(-admission_drag)),
            ("NET_PNL_CONTROL", Fraction(0)),
        ),
    )
    journal.commit_disposition(disposition, disposition.settlement_arrival_ns)
    return disposition


def _reduce_arm_events(
    data: ReferenceInput,
    arm: str,
    common_by_ordinal: Mapping[int, Mapping[str, Any] | None],
    journal: _Journal,
) -> ArmReplay:
    """Reduce causal clocks into journal-committed economic events."""
    terms = data.arms[arm]
    period_start = data.evaluation["period_start_ts_ns"]
    period_end = data.evaluation["period_end_ts_ns"]
    terminal_arrival = period_end - 1
    planned_entries: defaultdict[
        str,
        list[tuple[Proposal, str, Mapping[str, Any], MarketTick]],
    ] = defaultdict(list)
    terminal_rejections: dict[int, tuple[str, Mapping[str, Any] | None]] = {}
    signal_ids: dict[int, str] = {}
    for proposal in data.proposals:
        signal_id = _signal_id(data, proposal)
        signal_ids[proposal.ordinal] = signal_id
        common = common_by_ordinal[proposal.ordinal]
        if common is None:
            terminal_rejections[proposal.ordinal] = (
                "NO_COMMON_CAUSAL_PATH",
                None,
            )
            continue
        entry = _first_fill_tick(data, proposal, terms.latency_ns)
        if entry is None:
            terminal_rejections[proposal.ordinal] = ("NO_CAUSAL_FILL", common)
            continue
        planned_entries[entry.source_event_sha256].append(
            (proposal, signal_id, common, entry)
        )

    active: list[PositionLot] = []
    accepted_positions: list[PositionLot] = []
    closed_dispositions: list[ClosedDisposition] = []
    committed: dict[int, Disposition] = {}
    risk_timeline: list[RiskSnapshot] = []
    pending_rejections: defaultdict[int, list[PendingRejection]] = defaultdict(list)
    boundary_equities: dict[int, int] = {}
    halted = False

    def record_rejection(
        proposal: Proposal,
        signal_id: str,
        reason: str,
        common: Mapping[str, Any] | None,
        known_arrival_ns: int,
    ) -> None:
        gross = 0 if common is None else common["gross_pnl_jpy_micros"]
        settlement_arrival_ns = known_arrival_ns
        if gross and common is not None:
            settlement_arrival_ns = max(
                known_arrival_ns,
                common["exit_valuation_arrival_ns"],
            )
        pending = PendingRejection(
            arm=arm,
            proposal=proposal,
            signal_id=signal_id,
            economic_lot_id=_economic_lot_id(data, proposal),
            reason=reason,
            common=common,
            known_arrival_ns=known_arrival_ns,
            settlement_arrival_ns=settlement_arrival_ns,
        )
        if settlement_arrival_ns == known_arrival_ns:
            committed[proposal.ordinal] = _settle_rejection(journal, pending)
        else:
            pending_rejections[settlement_arrival_ns].append(pending)

    def flush_rejections(arrival_ns: int) -> None:
        for pending in sorted(
            pending_rejections.pop(arrival_ns, ()),
            key=lambda item: item.proposal.ordinal,
        ):
            disposition = _settle_rejection(journal, pending)
            committed[pending.proposal.ordinal] = disposition

    def close_due(tick: MarketTick) -> None:
        due_positions = [
            position for position in active
            if position.proposal.instrument == tick.instrument
            and tick.arrival_ts_ns >= position.due_arrival_ns
            and tick.source_ts_ns >= position.entry.source_ts_ns
        ]
        for position in sorted(due_positions, key=lambda item: item.proposal.ordinal):
            if not _is_trade_fresh(
                tick,
                position.due_arrival_ns,
                data.max_trade_quote_staleness_ns,
            ):
                raise ReferenceError("scheduled close quote stale")
            disposition = _settle_position(
                data,
                journal,
                position,
                tick,
                "FINITE_MAX_AGE",
            )
            position.closed_disposition = disposition
            committed[position.proposal.ordinal] = disposition
            closed_dispositions.append(disposition)
            active.remove(position)

    def closeout_all(arrival_ns: int, reason: str) -> None:
        watermark = _arrival_watermark(data, arrival_ns)
        frozen = [
            (
                position,
                _latest_fresh_tick(
                    data,
                    position.proposal.instrument,
                    watermark,
                    arrival_ns,
                    data.max_trade_quote_staleness_ns,
                ),
            )
            for position in sorted(active, key=lambda item: item.proposal.ordinal)
        ]
        for position, quote in frozen:
            disposition = _settle_position(
                data,
                journal,
                position,
                quote,
                reason,
                valuation_arrival_ns=arrival_ns,
                valuation_source_watermark_ns=watermark,
            )
            position.closed_disposition = disposition
            committed[position.proposal.ordinal] = disposition
            closed_dispositions.append(disposition)
            active.remove(position)

    period_ticks = tuple(
        tick for tick in data.ticks
        if period_start <= tick.arrival_ts_ns < period_end
    )
    ticks_by_arrival: defaultdict[int, list[MarketTick]] = defaultdict(list)
    for tick in period_ticks:
        ticks_by_arrival[tick.arrival_ts_ns].append(tick)
    boundary_clocks = {terminal_arrival}
    for month in _intersecting_months(period_start, period_end):
        _, month_end = _month_bounds_ns(month)
        checkpoint = min(period_end, month_end) - 1
        if checkpoint >= period_start:
            boundary_clocks.add(checkpoint)
    attribution_clocks = {
        common["exit_valuation_arrival_ns"]
        for common in common_by_ordinal.values()
        if common is not None
        and period_start <= common["exit_valuation_arrival_ns"] < period_end
    }
    event_clocks = sorted({*ticks_by_arrival, *boundary_clocks, *attribution_clocks})
    for arrival_ns in event_clocks:
        batch = sorted(
            ticks_by_arrival.get(arrival_ns, ()),
            key=lambda tick: (tick.source_ts_ns, tick.provider_id, tick.sequence),
        )

        # Causal phase order is fixed: exits, marks/risk, forced closeout,
        # admissions, final boundary snapshot, then attribution settlement.
        for tick in batch:
            close_due(tick)
        watermark = _arrival_watermark(data, arrival_ns)
        for position in sorted(active, key=lambda item: item.proposal.ordinal):
            mark_tick = _latest_fresh_tick(
                data,
                position.proposal.instrument,
                watermark,
                arrival_ns,
                data.max_trade_quote_staleness_ns,
            )
            _journal_mark(data, journal, position, mark_tick, arrival_ns)
        mark = _mark_state(data, active, closed_dispositions, arrival_ns)
        risk_timeline.append(mark)
        closeout_reason = _risk_closeout_reason(data, mark)
        if closeout_reason is not None:
            if active:
                closeout_all(arrival_ns, closeout_reason)
            halted = True
            risk_timeline.append(
                _mark_state(data, active, closed_dispositions, arrival_ns)
            )

        plans = [
            plan
            for tick in batch
            for plan in planned_entries.get(tick.source_event_sha256, ())
        ]
        for proposal, signal_id, common, entry in sorted(
            plans,
            key=lambda item: item[0].ordinal,
        ):
            if halted:
                record_rejection(
                    proposal, signal_id, "ACCOUNT_HALTED", common, arrival_ns
                )
                continue
            if any(
                position.proposal.instrument == proposal.instrument
                for position in active
            ):
                record_rejection(
                    proposal,
                    signal_id,
                    "SAME_PAIR_COLLISION_REJECTED",
                    common,
                    arrival_ns,
                )
                continue
            entry_price, entry_num, entry_den = _execution_price(
                entry,
                proposal,
                terms,
                data.registry,
                opening=True,
            )
            units_micros = _sized_units(data, proposal, entry, entry_price)
            if units_micros == 0:
                record_rejection(
                    proposal, signal_id, "SIZE_ROUNDED_TO_ZERO", common, arrival_ns
                )
                continue
            entry_notional_exact = _position_notional_exact(
                data,
                proposal,
                units_micros,
                entry_price,
                entry.arrival_ts_ns,
                opening=True,
            )
            position = PositionLot(
                arm=arm,
                proposal=proposal,
                signal_id=signal_id,
                economic_lot_id=_economic_lot_id(data, proposal),
                common=common,
                entry=entry,
                entry_price=entry_price,
                entry_price_numerator=entry_num,
                entry_price_denominator=entry_den,
                units_micros=units_micros,
                entry_notional_exact=entry_notional_exact,
                entry_notional_rounded=_ceil_nonnegative(entry_notional_exact),
                due_arrival_ns=entry.arrival_ts_ns + proposal.max_age_ns,
                entry_commission_exact=(
                    entry_notional_exact
                    * terms.commission_ppm_per_side
                    / 1_000_000
                ),
            )
            tentative = [*active, position]
            if len(tentative) > data.inventory["max_open_positions"]:
                record_rejection(
                    proposal, signal_id, "POSITION_CAP_REJECTED", common, arrival_ns
                )
                continue
            tentative_mark = _mark_state(
                data,
                tentative,
                closed_dispositions,
                arrival_ns,
            )
            gross = tentative_mark.gross_notional_jpy_micros
            exposure = tentative_mark.signed_currency_exposure_jpy_micros
            if gross > data.inventory["max_gross_notional_jpy_micros"]:
                record_rejection(
                    proposal, signal_id, "GROSS_CAP_REJECTED", common, arrival_ns
                )
                continue
            if max((abs(value) for value in exposure.values()), default=0) \
                    > data.inventory["max_currency_notional_jpy_micros"]:
                record_rejection(
                    proposal,
                    signal_id,
                    "CURRENCY_CAP_REJECTED",
                    common,
                    arrival_ns,
                )
                continue
            if _risk_closeout_reason(data, tentative_mark) == "MARGIN_CLOSEOUT":
                record_rejection(
                    proposal, signal_id, "MARGIN_ENTRY_REJECTED", common, arrival_ns
                )
                continue
            position.signed_exposure_after_entry = dict(exposure)
            position.gross_after_entry = gross
            position.marked_equity_after_entry = (
                tentative_mark.marked_equity_jpy_micros
            )
            position.required_margin_after_entry = (
                tentative_mark.required_margin_jpy_micros
            )
            position.free_margin_after_entry = tentative_mark.free_margin_jpy_micros
            _journal_entry(data, journal, position)
            _journal_mark(data, journal, position, entry, arrival_ns)
            active.append(position)
            accepted_positions.append(position)
            risk_timeline.append(tentative_mark)

        final_snapshot = _mark_state(
            data,
            active,
            closed_dispositions,
            arrival_ns,
        )
        if risk_timeline[-1] != final_snapshot:
            risk_timeline.append(final_snapshot)
        if arrival_ns in boundary_clocks:
            boundary_equities[arrival_ns] = final_snapshot.marked_equity_jpy_micros
        flush_rejections(arrival_ns)

    if active:
        terminal_watermark = _arrival_watermark(data, terminal_arrival)
        frozen_terminal = [
            (position, _terminal_tick(data, position.proposal.instrument))
            for position in sorted(active, key=lambda item: item.proposal.ordinal)
        ]
        for position, terminal in frozen_terminal:
            if terminal.arrival_ts_ns < position.entry.arrival_ts_ns:
                raise ReferenceError("terminal quote precedes entry")
        # The terminal clock is part of ``event_clocks``.  Its risk phase has
        # already marked every pre-existing position, and a position admitted
        # at that clock is marked before it enters ``active``.  Do not invoke a
        # second nominal mark here: the frozen close phase must begin only
        # after the one all-position mark phase is complete.
        pre_liquidation = _mark_state(
            data,
            active,
            closed_dispositions,
            terminal_arrival,
        )
        risk_timeline.append(pre_liquidation)
        terminal_reason = _risk_closeout_reason(
            data, pre_liquidation
        ) or "TERMINAL_LIQUIDATION"
        halted = halted or terminal_reason in {
            "MARGIN_CLOSEOUT", "INVENTORY_CAP_CLOSEOUT"
        }
        for position, terminal in frozen_terminal:
            disposition = _settle_position(
                data,
                journal,
                position,
                terminal,
                terminal_reason,
                valuation_arrival_ns=terminal_arrival,
                valuation_source_watermark_ns=terminal_watermark,
            )
            position.closed_disposition = disposition
            committed[position.proposal.ordinal] = disposition
            closed_dispositions.append(disposition)
            active.remove(position)
        post_liquidation = _mark_state(
            data,
            active,
            closed_dispositions,
            terminal_arrival,
        )
        if (
            pre_liquidation.marked_equity_jpy_micros
            != post_liquidation.marked_equity_jpy_micros
        ):
            raise ReferenceError("terminal valuation changed while liquidating")
        risk_timeline.append(post_liquidation)
        boundary_equities[terminal_arrival] = (
            post_liquidation.marked_equity_jpy_micros
        )

    for proposal in data.proposals:
        if proposal.ordinal in committed:
            continue
        reason, common = terminal_rejections.get(
            proposal.ordinal,
            ("NO_CAUSAL_FILL", common_by_ordinal[proposal.ordinal]),
        )
        record_rejection(
            proposal,
            signal_ids[proposal.ordinal],
            reason,
            common,
            terminal_arrival,
        )
    flush_rejections(terminal_arrival)
    if pending_rejections:
        raise ReferenceError("unresolved rejection attribution journal")
    expected_ordinals = {proposal.ordinal for proposal in data.proposals}
    if set(committed) != expected_ordinals:
        raise ReferenceError("proposal disposition cardinality mismatch")
    journal_ordinals = {
        disposition.position.proposal.ordinal
        if isinstance(disposition, ClosedDisposition)
        else disposition.proposal.ordinal
        for disposition in journal.arm_dispositions(arm)
    }
    if journal_ordinals != expected_ordinals:
        raise ReferenceError("journal disposition set mismatch")
    if active:
        raise ReferenceError("terminal inventory not empty")
    return ArmReplay(
        positions=tuple(accepted_positions),
        risk_snapshots=tuple(risk_timeline),
        boundary_equities=MappingProxyType(dict(sorted(boundary_equities.items()))),
    )


def _cluster_metrics_from_events(
    dispositions: Sequence[Disposition],
    evaluation: Mapping[str, Any],
) -> tuple[int, int, str, list[dict[str, Any]]]:
    by_bucket: defaultdict[int, list[ClosedDisposition]] = defaultdict(list)
    for disposition in dispositions:
        if isinstance(disposition, ClosedDisposition):
            by_bucket[
                disposition.position.entry.arrival_ts_ns
                // evaluation["cluster_window_ns"]
            ].append(disposition)
    observations: list[dict[str, Any]] = []
    exact_returns: list[tuple[Fraction, str]] = []
    initial_equity = evaluation["initial_equity_jpy_micros"]
    for bucket, bucket_records in sorted(by_bucket.items()):
        parent: dict[str, str] = {}

        def find(currency: str) -> str:
            parent.setdefault(currency, currency)
            while parent[currency] != currency:
                parent[currency] = parent[parent[currency]]
                currency = parent[currency]
            return currency

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for disposition in bucket_records:
            union(*_pair(disposition.position.proposal.instrument))
        components: defaultdict[str, list[ClosedDisposition]] = defaultdict(list)
        for disposition in bucket_records:
            instrument = disposition.position.proposal.instrument
            components[find(_pair(instrument)[0])].append(disposition)
        for component in components.values():
            nodes = sorted({
                currency
                for disposition in component
                for currency in _pair(disposition.position.proposal.instrument)
            })
            exact_pnl = sum(
                (
                    disposition.values["economic_net_exact"]
                    for disposition in component
                ),
                Fraction(),
            )
            risk_pnl = _floor_fraction(exact_pnl)
            ledger_pnl = sum(
                disposition.values["net_pnl_jpy_micros"]
                for disposition in component
            )
            cluster_id = sha256_bytes(canonical_bytes({
                "time_bucket": bucket,
                "currency_nodes": nodes,
            }))
            observations.append({
                "cluster_id": cluster_id,
                "time_bucket": bucket,
                "currency_nodes": nodes,
                "source_signal_set_sha256": sha256_bytes(canonical_bytes(sorted({
                    disposition.position.economic_lot_id
                    for disposition in component
                }))),
                "ledger_net_pnl_jpy_micros": ledger_pnl,
                "cluster_risk_net_pnl_jpy_micros": risk_pnl,
                "signed_return": _ratio_text(Fraction(risk_pnl, initial_equity)),
            })
            exact_returns.append((exact_pnl, cluster_id))
    observations.sort(key=lambda row: row["cluster_id"])
    ordered = sorted(exact_returns, key=lambda item: (item[0], item[1]))
    tail_count = (
        max(1, (len(ordered) * evaluation["cvar_tail_bps"] + 9_999) // 10_000)
        if ordered else 0
    )
    if not tail_count:
        return 0, 0, "0.000000000000000000", observations
    tail_total = sum((item[0] for item in ordered[:tail_count]), Fraction())
    cvar_exact = tail_total / tail_count
    return (
        len(observations),
        _floor_fraction(cvar_exact),
        _ratio_text(cvar_exact / initial_equity),
        observations,
    )


def _journal_integral(
    account: Mapping[str, Fraction],
    name: str,
) -> int:
    value = account.get(name, Fraction())
    if value.denominator != 1:
        raise ReferenceError(f"journal control account is fractional: {name}")
    return value.numerator


def _derive_arm_metrics(
    data: ReferenceInput,
    arm: str,
    dispositions: Sequence[Disposition],
    replay: ArmReplay,
    journal: _Journal,
) -> dict[str, Any]:
    initial = data.evaluation["initial_equity_jpy_micros"]
    filled = [
        disposition
        for disposition in dispositions
        if isinstance(disposition, ClosedDisposition)
    ]
    rejected = [
        disposition
        for disposition in dispositions
        if isinstance(disposition, RejectedDisposition)
    ]
    balances = defaultdict(
        Fraction,
        journal.balances().get(arm, {}),
    )
    gross = _journal_integral(balances, "COMMON_GROSS_REFERENCE")
    sizing_drag = -_journal_integral(balances, "FILL_SIZING_DRAG")
    latency_drag = -_journal_integral(
        balances,
        "LATENCY_SPREAD_SLIPPAGE_DRAG",
    )
    direct_cost = -_journal_integral(balances, "DIRECT_COST")
    admission_drag = -_journal_integral(
        balances,
        "ADMISSION_OPPORTUNITY_DRAG",
    )
    net = -_journal_integral(balances, "NET_PNL_CONTROL")
    realized_cost = sum(
        disposition.common_gross_jpy_micros
        - disposition.values["net_pnl_jpy_micros"]
        for disposition in filled
    )
    decomposed = sizing_drag + latency_drag + direct_cost + admission_drag
    if decomposed != gross - net:
        raise ReferenceError("economic attribution does not reconcile")

    monthly: list[dict[str, Any]] = []
    start = data.evaluation["period_start_ts_ns"]
    end = data.evaluation["period_end_ts_ns"]
    previous_end_equity = initial
    for month in _intersecting_months(start, end):
        month_start, month_end = _month_bounds_ns(month)
        segment_start = max(start, month_start)
        segment_end = min(end, month_end)
        if segment_start == start:
            start_equity = initial
        else:
            try:
                start_equity = replay.boundary_equities[segment_start - 1]
            except KeyError as error:
                raise ReferenceError("missing month-start reducer snapshot") from error
            if start_equity != previous_end_equity:
                raise ReferenceError("month boundary equity is discontinuous")
        try:
            end_equity = replay.boundary_equities[segment_end - 1]
        except KeyError as error:
            raise ReferenceError("missing month-end reducer snapshot") from error
        multiple_defined = start_equity > 0
        monthly.append({
            "month_id": month,
            "comparable_full_month": month_start >= start and month_end <= end,
            "segment_start_ts_ns": segment_start,
            "segment_end_ts_ns": segment_end,
            "start_equity_jpy_micros": start_equity,
            "end_equity_jpy_micros": end_equity,
            "equity_multiple": (
                _ratio_text(Fraction(end_equity, start_equity))
                if multiple_defined else None
            ),
            "equity_multiple_status": (
                "DEFINED" if multiple_defined else "UNDEFINED_NONPOSITIVE_START_EQUITY"
            ),
            "ruin_observed": start_equity <= 0 or end_equity <= 0,
        })
        previous_end_equity = end_equity
    peak = initial
    max_drawdown = 0
    max_drawdown_ratio = Fraction()
    observations = [
        (mark.arrival_ts_ns, index, mark.marked_equity_jpy_micros)
        for index, mark in enumerate(replay.risk_snapshots)
    ]
    for _, _, equity in sorted(observations):
        peak = max(peak, equity)
        drawdown = peak - equity
        ratio = Fraction(drawdown, peak) if peak > 0 else Fraction(1)
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_ratio = max(max_drawdown_ratio, ratio)
    n_eff, cvar, cvar_return, clusters = _cluster_metrics_from_events(
        dispositions,
        data.evaluation,
    )
    max_gross = max(
        (mark.gross_notional_jpy_micros for mark in replay.risk_snapshots),
        default=0,
    )
    min_equity = min([
        initial,
        *(mark.marked_equity_jpy_micros for mark in replay.risk_snapshots),
        *replay.boundary_equities.values(),
    ])
    max_required = max(
        (mark.required_margin_jpy_micros for mark in replay.risk_snapshots),
        default=0,
    )
    min_free = min(
        (mark.free_margin_jpy_micros for mark in replay.risk_snapshots),
        default=initial,
    )
    statuses = [
        "FILLED_CLOSED"
        if isinstance(disposition, ClosedDisposition)
        else disposition.reason
        for disposition in dispositions
    ]
    signal_ids = [
        disposition.position.signal_id
        if isinstance(disposition, ClosedDisposition)
        else disposition.signal_id
        for disposition in dispositions
    ]
    return {
        "proposal_count": len(dispositions),
        "executed_count": len(filled),
        "disposition_counts": dict(sorted(Counter(statuses).items())),
        "signal_id_set_sha256": sha256_bytes(canonical_bytes(sorted(
            signal_ids
        ))),
        "common_gross_pnl_jpy_micros": gross,
        "realized_cost_jpy_micros": realized_cost,
        "fill_sizing_drag_jpy_micros": sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": latency_drag,
        "direct_commission_financing_cost_jpy_micros": direct_cost,
        "admission_opportunity_drag_jpy_micros": admission_drag,
        "total_execution_and_admission_drag_jpy_micros": decomposed,
        "net_pnl_jpy_micros": net,
        "ending_equity_jpy_micros": initial + net,
        "ending_equity_multiple": _ratio_text(Fraction(initial + net, initial)),
        "direction_accuracy": (
            _ratio_text(Fraction(
                sum(
                    disposition.common_gross_jpy_micros > 0
                    for disposition in filled
                ),
                len(filled),
            ))
            if filled else "0.000000000000000000"
        ),
        "max_drawdown_jpy_micros": max_drawdown,
        "max_drawdown_ratio": _ratio_text(
            max_drawdown_ratio,
            outward_nonnegative=True,
        ),
        "cvar_tail_bps": data.evaluation["cvar_tail_bps"],
        "cluster_cvar_jpy_micros": cvar,
        "cluster_cvar_return": cvar_return,
        "currency_time_cluster_n_eff": n_eff,
        "currency_time_cluster_observations": clusters,
        "monthly": monthly,
        "max_gross_notional_jpy_micros": max_gross,
        "minimum_marked_equity_jpy_micros": min_equity,
        "maximum_required_margin_jpy_micros": max_required,
        "minimum_free_margin_jpy_micros": min_free,
        "margin_guard_pass": (
            min_equity > 0
            and min_free >= 0
            and max_gross <= data.evaluation["margin_notional_cap_jpy_micros"]
            and all(mark.margin_ratio_pass is True for mark in replay.risk_snapshots)
            and all(
                disposition.exit_reason
                not in {"MARGIN_CLOSEOUT", "INVENTORY_CAP_CLOSEOUT"}
                for disposition in filled
            )
        ),
        "terminal_open_positions": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }


def _ledger_chain(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for sequence, source in enumerate(rows, 1):
        row = {
            "ledger_schema_version": 2,
            "ledger_sequence": sequence,
            "previous_hash": previous,
            **dict(source),
        }
        row["record_hash"] = _embedded_hash(row, "record_hash")
        result.append(row)
        previous = row["record_hash"]
    return result


def _validate_journal_reconciliation(
    journal: _Journal,
    dispositions_by_arm: Mapping[str, Sequence[Disposition]],
) -> None:
    balances = journal.balances()
    for arm in ARMS:
        account = defaultdict(Fraction, balances.get(arm, {}))
        for clearing_account in (
            "POSITION_BASIS",
            "POSITION_CONTROL",
            "UNREALIZED_ASSET",
            "UNREALIZED_PNL",
        ):
            if account[clearing_account] != 0:
                raise ReferenceError("journal terminal position balance is nonzero")
        dispositions = dispositions_by_arm[arm]
        filled = [
            disposition
            for disposition in dispositions
            if isinstance(disposition, ClosedDisposition)
        ]
        rejected = [
            disposition
            for disposition in dispositions
            if isinstance(disposition, RejectedDisposition)
        ]
        exact_net = sum(
            (
                disposition.values["economic_net_exact"]
                for disposition in filled
            ),
            Fraction(),
        )
        if account["SETTLEMENT_CASH"] != exact_net:
            raise ReferenceError("position journal does not reconcile exact economic net")
        ledger_net = sum(
            disposition.values["net_pnl_jpy_micros"]
            for disposition in filled
        )
        if account["NET_PNL_CONTROL"] != -ledger_net:
            raise ReferenceError("attribution journal does not reconcile ledger net")
        rejected_amounts = {
            disposition.proposal.ordinal: _rejection_amounts(disposition)
            for disposition in rejected
        }
        expected_attribution = {
            "COMMON_GROSS_REFERENCE": (
                sum(
                    disposition.common_gross_jpy_micros
                    for disposition in filled
                )
                + sum(amounts[0] for amounts in rejected_amounts.values())
            ),
            "FILL_SIZING_DRAG": -(
                sum(
                    disposition.fill_sizing_drag_jpy_micros
                    for disposition in filled
                )
                + sum(amounts[1] for amounts in rejected_amounts.values())
            ),
            "LATENCY_SPREAD_SLIPPAGE_DRAG": -(
                sum(
                    disposition.execution_drag_jpy_micros
                    for disposition in filled
                )
                + sum(amounts[2] for amounts in rejected_amounts.values())
            ),
            "DIRECT_COST": -sum(
                disposition.values["commission_jpy_micros"]
                + disposition.values["financing_jpy_micros"]
                for disposition in filled
            ),
            "ADMISSION_OPPORTUNITY_DRAG": -sum(
                amounts[3] for amounts in rejected_amounts.values()
            ),
        }
        for account_name, expected in expected_attribution.items():
            if account[account_name] != expected:
                raise ReferenceError("attribution journal account mismatch")


def replay_reference(artifacts: Mapping[str, bytes]) -> dict[str, Any]:
    """Replay the frozen economic inputs through the independent journal."""
    data = decode_reference_input(artifacts)
    common_by_ordinal = {
        proposal.ordinal: _common_reference(data, proposal)
        for proposal in data.proposals
    }
    journal = _Journal()
    replay_by_arm: dict[str, ArmReplay] = {}
    for arm in ARMS:
        replay_by_arm[arm] = _reduce_arm_events(
            data,
            arm,
            common_by_ordinal,
            journal,
        )
    dispositions_by_arm = {
        arm: journal.arm_dispositions(arm)
        for arm in ARMS
    }
    _validate_journal_reconciliation(journal, dispositions_by_arm)
    signal_sets = {
        arm: sorted(
            disposition.position.signal_id
            if isinstance(disposition, ClosedDisposition)
            else disposition.signal_id
            for disposition in dispositions_by_arm[arm]
        )
        for arm in ARMS
    }
    if len({tuple(values) for values in signal_sets.values()}) != 1:
        raise ReferenceError("signal identities diverge across arms")
    ordered_dispositions = sorted(
        (
            disposition
            for arm in ARMS
            for disposition in dispositions_by_arm[arm]
        ),
        key=lambda disposition: (
            disposition.position.proposal.ordinal
            if isinstance(disposition, ClosedDisposition)
            else disposition.proposal.ordinal,
            ARMS.index(
                disposition.position.arm
                if isinstance(disposition, ClosedDisposition)
                else disposition.arm
            ),
        ),
    )
    ordered_records = [
        _project_closed_disposition(data, disposition)
        if isinstance(disposition, ClosedDisposition)
        else _project_rejected_disposition(disposition)
        for disposition in ordered_dispositions
    ]
    ledger_rows = _ledger_chain(ordered_records)
    ledger_bytes = b"".join(canonical_bytes(row) + b"\n" for row in ledger_rows)
    arm_metrics = {
        arm: _derive_arm_metrics(
            data,
            arm,
            dispositions_by_arm[arm],
            replay_by_arm[arm],
            journal,
        )
        for arm in ARMS
    }
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "initial_equity_jpy_micros": data.evaluation["initial_equity_jpy_micros"],
        "same_signal_ids_all_arms": True,
        "all_proposals_have_all_arm_dispositions": all(
            len(dispositions_by_arm[arm]) == len(data.proposals) for arm in ARMS
        ),
        "common_gross_reference_shared": all(
            len({
                (
                    disposition.common_gross_jpy_micros
                    if isinstance(disposition, ClosedDisposition)
                    else _rejection_amounts(disposition)[0]
                )
                for disposition in ordered_dispositions
                if (
                    disposition.position.proposal.ordinal
                    if isinstance(disposition, ClosedDisposition)
                    else disposition.proposal.ordinal
                ) == proposal.ordinal
            }) == 1
            for proposal in data.proposals
        ),
        "arms": arm_metrics,
        "external_orders": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }
    metrics["metrics_sha256"] = _embedded_hash(metrics, "metrics_sha256")
    proposal_provenance_root = sha256_bytes(canonical_bytes({
        "provenance": dict(data.provenance),
        "rows": [
            {
                "proposal_ordinal": proposal.ordinal,
                "decision_source_event_sha256": proposal.decision_source_event_sha256,
                "completed_data_watermark_source_ts_ns": (
                    proposal.completed_data_watermark_source_ts_ns
                ),
                "completed_data_prefix_root_sha256": (
                    proposal.completed_data_prefix_root_sha256
                ),
            }
            for proposal in data.proposals
        ],
    }))
    input_root = sha256_bytes(canonical_bytes({
        "artifact_sha256": dict(sorted(data.raw_hashes.items())),
    }))
    ledger_sha256 = sha256_bytes(ledger_bytes)
    ledger_terminal_hash = (
        ledger_rows[-1]["record_hash"] if ledger_rows else ZERO_SHA256
    )
    journal_root = journal.root()
    journal_transaction_count = len(journal.transactions)
    projection = {
        "all_transactions_balanced": True,
        "engine_id": ENGINE_ID,
        "input_root_sha256": input_root,
        "journal_root_sha256": journal_root,
        "journal_transaction_count": journal_transaction_count,
        "ledger_row_count": len(ledger_rows),
        "ledger_sha256": ledger_sha256,
        "ledger_terminal_hash": ledger_terminal_hash,
        "oracle_metrics_sha256": metrics["metrics_sha256"],
        "proposal_provenance_root_sha256": proposal_provenance_root,
    }
    return {
        "engine_id": ENGINE_ID,
        "input_root_sha256": input_root,
        "ledger_bytes": ledger_bytes,
        "ledger_row_count": len(ledger_rows),
        "ledger_terminal_hash": ledger_terminal_hash,
        "oracle_metrics": metrics,
        "proposal_provenance_root_sha256": proposal_provenance_root,
        "journal_root_sha256": journal_root,
        "journal_transaction_count": journal_transaction_count,
        "all_transactions_balanced": True,
        "economic_projection_sha256": sha256_bytes(canonical_bytes(projection)),
    }
