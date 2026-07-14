from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping


G8_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")

# Default watchlist for the 5-minute trader cycle.
#
# Market reality: the old 7-pair list covered mostly USD majors and JPY crosses,
# while the currency-strength layer already reasons over the full G8 28-pair
# matrix. A 5-minute campaign that needs many small, bounded attempts should
# watch the same G8 universe it uses for strength and relative-value context.
# This is a coverage universe, not permission to trade every pair; missing
# OANDA instruments, stale quotes, spreads, calendar windows, profile status,
# and risk validation still block individual lanes.
DEFAULT_TRADER_PAIRS: tuple[str, ...] = (
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CAD", "USD_CHF",
    "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_NZD",
    "GBP_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
)

DEFAULT_TRADER_PAIRS_ARG = ",".join(DEFAULT_TRADER_PAIRS)

# Non-FX market context instruments. These are monitored with the same
# multi-timeframe technical stack as FX, but they are not automatically tradeable:
# broker account instruments must explicitly list them before any order path can
# treat them as candidates.
DEFAULT_CONTEXT_ASSETS: tuple[str, ...] = (
    # Equity indices
    "SPX500_USD", "NAS100_USD", "US30_USD", "JP225_USD", "DE30_EUR", "UK100_GBP",
    # Bonds
    "USB02Y_USD", "USB05Y_USD", "USB10Y_USD", "USB30Y_USD",
    # Commodities
    "XAU_USD", "XAG_USD", "BCO_USD", "WTICO_USD", "NATGAS_USD",
    # Crypto
    "BTC_USD", "ETH_USD",
)

DEFAULT_CONTEXT_ASSETS_ARG = ",".join(DEFAULT_CONTEXT_ASSETS)

OANDA_SPREAD_CALIBRATION_V1_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "config" / "oanda_spread_calibration_v1.json"
)
OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256: Final[str] = (
    "f8158be138ee641f43dbf299da94b9d6704a88e099440ec163391d16ebe1984f"
)
OANDA_SPREAD_CALIBRATION_V1_SOURCE_EVIDENCE_SHA256: Final[str] = (
    "9a42bc509ca6c6dfe868c5504a80a516c9f9bfe861925c329b368676e3d10d51"
)

_CALIBRATION_SCHEMA = "QR_OANDA_SPREAD_CALIBRATION_V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_CALIBRATION_BYTES = 64 * 1024
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "calibration_sha256",
        "source_evidence_sha256",
        "evidence_policy_version",
        "max_age_days_after_window",
        "valid_until_utc",
        "method",
        "window",
        "business_days_utc",
        "session",
        "broker_http_methods_used",
        "broker_write_performed",
        "pairs",
    }
)
_METHOD_KEYS = frozenset(
    {
        "broker",
        "candle_price_component",
        "granularity",
        "spread_definition",
        "excluded_price_fields",
        "percentile_method",
        "recommended_baseline_formula",
        "max_spread_multiple",
        "rounding_step_pips",
    }
)
_PAIR_KEYS = frozenset(
    {
        "pair",
        "sample_count",
        "p50_pips",
        "p95_pips",
        "p99_pips",
        "max_pips",
        "recommended_baseline_pips",
    }
)


class SpreadCalibrationError(RuntimeError):
    """Raised when the pinned broker-spread calibration is unavailable or invalid."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpreadCalibrationPair:
    pair: str
    sample_count: int
    p50_pips: float
    p95_pips: float
    p99_pips: float
    max_pips: float
    recommended_baseline_pips: float


@dataclass(frozen=True, slots=True)
class OandaSpreadCalibrationV1:
    bytes_sha256: str
    calibration_sha256: str
    source_evidence_sha256: str
    evidence_policy_version: str
    max_age_days_after_window: int
    window_to_utc: datetime
    valid_until_utc: datetime
    business_days_utc: tuple[str, ...]
    pairs: Mapping[str, SpreadCalibrationPair]


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_json_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _require_object(
    value: object,
    *,
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpreadCalibrationError(f"{label} must be an object")
    result = value
    if frozenset(result) != expected_keys:
        raise SpreadCalibrationError(f"{label} has non-canonical keys")
    return result


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpreadCalibrationError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpreadCalibrationError(f"{label} must be an integer")
    return value


def _require_positive_float(value: object, *, label: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
        raise SpreadCalibrationError(f"{label} must be a positive finite JSON float")
    return value


def _canonical_content_sha256(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("calibration_sha256", None)
    try:
        encoded = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpreadCalibrationError("calibration canonicalization failed") from exc
    return hashlib.sha256(encoded).hexdigest()


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    text = _require_string(value, label=label)
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise SpreadCalibrationError(f"{label} must be canonical UTC") from exc


def _parse_business_days(
    value: object,
    *,
    window_from: datetime,
    window_to: datetime,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise SpreadCalibrationError("business_days_utc must contain exactly 6 days")
    parsed_days: list[date] = []
    labels: list[str] = []
    for index, item in enumerate(value):
        text = _require_string(item, label=f"business_days_utc[{index}]")
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise SpreadCalibrationError(
                f"business_days_utc[{index}] must be YYYY-MM-DD"
            ) from exc
        if parsed.isoformat() != text or parsed.weekday() >= 5:
            raise SpreadCalibrationError("business_days_utc must be canonical weekdays")
        parsed_days.append(parsed)
        labels.append(text)
    if parsed_days != sorted(set(parsed_days)):
        raise SpreadCalibrationError("business_days_utc must be unique and sorted")

    expected_days: list[date] = []
    cursor = window_from.date()
    while cursor <= window_to.date():
        if cursor.weekday() < 5:
            expected_days.append(cursor)
        cursor += timedelta(days=1)
    if parsed_days != expected_days:
        raise SpreadCalibrationError("business_days_utc does not match the window")
    return tuple(labels)


def _validate_method(value: object) -> int:
    method = _require_object(value, expected_keys=_METHOD_KEYS, label="method")
    exact_strings = {
        "broker": "OANDA",
        "candle_price_component": "MBA",
        "granularity": "M5",
        "spread_definition": "max(ASK.o-BID.o, ASK.c-BID.c) in pips",
        "percentile_method": "nearest_rank",
        "recommended_baseline_formula": (
            "ceil(P95_endpoint_spread / RiskPolicy.max_spread_multiple, 0.1 pip)"
        ),
    }
    for key, expected in exact_strings.items():
        if _require_string(method[key], label=f"method.{key}") != expected:
            raise SpreadCalibrationError(f"method.{key} is not canonical")
    if not isinstance(method["excluded_price_fields"], list) or method[
        "excluded_price_fields"
    ] != ["h", "l"]:
        raise SpreadCalibrationError("method.excluded_price_fields is not canonical")
    multiple = _require_positive_float(
        method["max_spread_multiple"], label="method.max_spread_multiple"
    )
    rounding_step = _require_positive_float(
        method["rounding_step_pips"], label="method.rounding_step_pips"
    )
    if Decimal(str(multiple)) != Decimal("2.5"):
        raise SpreadCalibrationError("method.max_spread_multiple must remain 2.5")
    if Decimal(str(rounding_step)) != Decimal("0.1"):
        raise SpreadCalibrationError("method.rounding_step_pips must remain 0.1")
    return 5


def _validate_session(value: object) -> tuple[int, int]:
    session = _require_object(
        value,
        expected_keys=frozenset(
            {
                "timezone",
                "start_hour_inclusive",
                "end_hour_exclusive",
                "weekday_filter",
            }
        ),
        label="session",
    )
    if _require_string(session["timezone"], label="session.timezone") != "UTC":
        raise SpreadCalibrationError("session.timezone must be UTC")
    if (
        _require_string(session["weekday_filter"], label="session.weekday_filter")
        != "MONDAY_TO_FRIDAY"
    ):
        raise SpreadCalibrationError("session.weekday_filter is not canonical")
    start = _require_int(
        session["start_hour_inclusive"], label="session.start_hour_inclusive"
    )
    end = _require_int(
        session["end_hour_exclusive"], label="session.end_hour_exclusive"
    )
    if (start, end) != (12, 15):
        raise SpreadCalibrationError("session hours must remain 12<=UTC<15")
    return start, end


def _validate_pairs(
    value: object,
    *,
    expected_sample_count: int,
) -> Mapping[str, SpreadCalibrationPair]:
    if not isinstance(value, list) or len(value) != len(DEFAULT_TRADER_PAIRS):
        raise SpreadCalibrationError("pairs must contain the exact 28-pair universe")
    calibrated: dict[str, SpreadCalibrationPair] = {}
    ordered_pairs: list[str] = []
    for index, item in enumerate(value):
        row = _require_object(
            item,
            expected_keys=_PAIR_KEYS,
            label=f"pairs[{index}]",
        )
        pair = _require_string(row["pair"], label=f"pairs[{index}].pair")
        sample_count = _require_int(
            row["sample_count"], label=f"pairs[{index}].sample_count"
        )
        if sample_count != expected_sample_count:
            raise SpreadCalibrationError(
                f"{pair} sample_count must equal {expected_sample_count}"
            )
        p50 = _require_positive_float(row["p50_pips"], label=f"{pair}.p50_pips")
        p95 = _require_positive_float(row["p95_pips"], label=f"{pair}.p95_pips")
        p99 = _require_positive_float(row["p99_pips"], label=f"{pair}.p99_pips")
        maximum = _require_positive_float(row["max_pips"], label=f"{pair}.max_pips")
        baseline = _require_positive_float(
            row["recommended_baseline_pips"],
            label=f"{pair}.recommended_baseline_pips",
        )
        if not p50 <= p95 <= p99 <= maximum:
            raise SpreadCalibrationError(f"{pair} percentile ordering is invalid")
        derived = (Decimal(str(p95)) / Decimal("2.5")).quantize(
            Decimal("0.1"), rounding=ROUND_CEILING
        )
        if Decimal(str(baseline)) != derived:
            raise SpreadCalibrationError(
                f"{pair} recommended baseline must equal ceil(p95/2.5, 0.1 pip)"
            )
        ordered_pairs.append(pair)
        calibrated[pair] = SpreadCalibrationPair(
            pair=pair,
            sample_count=sample_count,
            p50_pips=p50,
            p95_pips=p95,
            p99_pips=p99,
            max_pips=maximum,
            recommended_baseline_pips=baseline,
        )
    if tuple(ordered_pairs) != DEFAULT_TRADER_PAIRS or len(calibrated) != len(
        DEFAULT_TRADER_PAIRS
    ):
        raise SpreadCalibrationError("pairs are not the canonical ordered G8 universe")
    return MappingProxyType(calibrated)


def load_oanda_spread_calibration_v1(
    path: Path,
    *,
    expected_bytes_sha256: str,
    evaluated_at_utc: datetime,
) -> OandaSpreadCalibrationV1:
    """Load one pinned, GET-only broker calibration or fail closed."""

    if not isinstance(expected_bytes_sha256, str) or _SHA256_RE.fullmatch(
        expected_bytes_sha256
    ) is None:
        raise SpreadCalibrationError("expected calibration bytes SHA-256 is invalid")
    if (
        evaluated_at_utc.__class__ is not datetime
        or evaluated_at_utc.tzinfo is None
        or evaluated_at_utc.utcoffset() is None
    ):
        raise SpreadCalibrationError("evaluated_at_utc must be an aware datetime")
    evaluation_time = evaluated_at_utc.astimezone(timezone.utc)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SpreadCalibrationError(f"spread calibration is unavailable: {path}") from exc
    if not raw or len(raw) > _MAX_CALIBRATION_BYTES:
        raise SpreadCalibrationError("spread calibration byte length is invalid")
    bytes_sha256 = hashlib.sha256(raw).hexdigest()
    if bytes_sha256 != expected_bytes_sha256:
        raise SpreadCalibrationError("spread calibration bytes SHA-256 mismatch")
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKey, ValueError) as exc:
        raise SpreadCalibrationError("spread calibration JSON is invalid") from exc
    root = _require_object(payload, expected_keys=_TOP_LEVEL_KEYS, label="calibration")

    if _require_string(root["schema"], label="schema") != _CALIBRATION_SCHEMA:
        raise SpreadCalibrationError("spread calibration schema is invalid")
    calibration_sha256 = _require_string(
        root["calibration_sha256"], label="calibration_sha256"
    )
    if _SHA256_RE.fullmatch(calibration_sha256) is None:
        raise SpreadCalibrationError("calibration_sha256 is invalid")
    if calibration_sha256 != _canonical_content_sha256(root):
        raise SpreadCalibrationError("calibration_sha256 mismatch")
    source_evidence_sha256 = _require_string(
        root["source_evidence_sha256"], label="source_evidence_sha256"
    )
    if source_evidence_sha256 != OANDA_SPREAD_CALIBRATION_V1_SOURCE_EVIDENCE_SHA256:
        raise SpreadCalibrationError("source_evidence_sha256 is not the pinned evidence")
    evidence_policy_version = _require_string(
        root["evidence_policy_version"], label="evidence_policy_version"
    )
    if evidence_policy_version != "OANDA_M5_MBA_SESSION_SPREAD_MONTHLY_V1":
        raise SpreadCalibrationError("evidence_policy_version is not canonical")
    max_age_days = _require_int(
        root["max_age_days_after_window"], label="max_age_days_after_window"
    )
    if max_age_days != 31:
        raise SpreadCalibrationError("max_age_days_after_window must remain 31")

    granularity_minutes = _validate_method(root["method"])
    window = _require_object(
        root["window"],
        expected_keys=frozenset({"from_utc", "to_utc"}),
        label="window",
    )
    window_from = _parse_utc_timestamp(window["from_utc"], label="window.from_utc")
    window_to = _parse_utc_timestamp(window["to_utc"], label="window.to_utc")
    if window_from >= window_to:
        raise SpreadCalibrationError("calibration window must be positive")
    valid_until = _parse_utc_timestamp(
        root["valid_until_utc"], label="valid_until_utc"
    )
    if valid_until != window_to + timedelta(days=max_age_days):
        raise SpreadCalibrationError(
            "valid_until_utc must equal window.to_utc plus the 31-day policy"
        )
    if evaluation_time < window_to:
        raise SpreadCalibrationError("spread calibration is not yet valid")
    if evaluation_time > valid_until:
        raise SpreadCalibrationError("spread calibration is expired")
    business_days = _parse_business_days(
        root["business_days_utc"],
        window_from=window_from,
        window_to=window_to,
    )
    session_start_hour, session_end_hour = _validate_session(root["session"])
    if (
        (
            window_from.hour,
            window_from.minute,
            window_from.second,
            window_from.microsecond,
        )
        != (session_start_hour, 0, 0, 0)
        or (
            window_to.hour,
            window_to.minute,
            window_to.second,
            window_to.microsecond,
        )
        != (session_end_hour, 0, 0, 0)
    ):
        raise SpreadCalibrationError("window boundaries must match the UTC session")
    session_minutes = (session_end_hour - session_start_hour) * 60
    if session_minutes % granularity_minutes:
        raise SpreadCalibrationError("session is not divisible into complete M5 samples")
    expected_sample_count = (
        len(business_days) * session_minutes // granularity_minutes
    )
    if expected_sample_count != 216:
        raise SpreadCalibrationError("window/session/granularity must imply 216 samples")
    if not isinstance(root["broker_http_methods_used"], list) or root[
        "broker_http_methods_used"
    ] != ["GET"]:
        raise SpreadCalibrationError("calibration evidence must be GET-only")
    if root["broker_write_performed"] is not False:
        raise SpreadCalibrationError("calibration evidence must prove no broker write")
    calibrated_pairs = _validate_pairs(
        root["pairs"],
        expected_sample_count=expected_sample_count,
    )
    return OandaSpreadCalibrationV1(
        bytes_sha256=bytes_sha256,
        calibration_sha256=calibration_sha256,
        source_evidence_sha256=source_evidence_sha256,
        evidence_policy_version=evidence_policy_version,
        max_age_days_after_window=max_age_days,
        window_to_utc=window_to,
        valid_until_utc=valid_until,
        business_days_utc=business_days,
        pairs=calibrated_pairs,
    )


# Broker-spec spread anomaly baselines used by RiskEngine's current-spread cap.
# The runtime table has no static fallback: missing, modified, or semantically
# invalid tracked bytes raise during import before any trading service can start.
OANDA_SPREAD_CALIBRATION_V1 = load_oanda_spread_calibration_v1(
    OANDA_SPREAD_CALIBRATION_V1_PATH,
    expected_bytes_sha256=OANDA_SPREAD_CALIBRATION_V1_BYTES_SHA256,
    evaluated_at_utc=datetime.now(timezone.utc),
)
NORMAL_SPREAD_PIPS_CALIBRATION_SHA256: Final[str] = (
    OANDA_SPREAD_CALIBRATION_V1.calibration_sha256
)
NORMAL_SPREAD_PIPS_SOURCE_EVIDENCE_SHA256: Final[str] = (
    OANDA_SPREAD_CALIBRATION_V1.source_evidence_sha256
)
NORMAL_SPREAD_PIPS: Mapping[str, float] = MappingProxyType(
    {
        pair: OANDA_SPREAD_CALIBRATION_V1.pairs[pair].recommended_baseline_pips
        for pair in DEFAULT_TRADER_PAIRS
    }
)


def instrument_pip_factor(pair: str) -> int:
    """Return the broker price precision used for one pip.

    JPY-quoted FX pairs conventionally quote one pip as 0.01; the other G8
    FX pairs quote one pip as 0.0001. This is a broker-spec convention, not a
    trading threshold.
    """

    return 100 if pair.upper().endswith("_JPY") else 10000
