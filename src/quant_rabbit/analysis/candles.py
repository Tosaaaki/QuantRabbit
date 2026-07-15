"""OANDA OHLC candle fetcher.

The trader needs current candles to read the chart. This module returns
plain-Python tuples of `Candle` records so downstream indicator code stays
dependency-free.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from quant_rabbit.broker.oanda import OandaReadOnlyClient


SUPPORTED_GRANULARITIES: frozenset[str] = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D"})
TECHNICAL_CANDLE_INTEGRITY_SCHEMA = "QR_TECHNICAL_CANDLE_INTEGRITY_V2"
PAIR_TECHNICAL_CANDLE_INTEGRITY_SCHEMA = "QR_PAIR_TECHNICAL_CANDLE_INTEGRITY_V2"
TECHNICAL_CANDLE_SPREAD_CONTAMINATED = "TECHNICAL_CANDLE_SPREAD_CONTAMINATED"
TECHNICAL_CANDLE_PROVENANCE_INVALID = "TECHNICAL_CANDLE_PROVENANCE_INVALID"
TECHNICAL_CANDLE_SPREAD_EXECUTION_MODE = "EXECUTION_ENDPOINT_CAP"
TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE = "PROVENANCE_ONLY_HIGHER_TIMEFRAME"
TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES: frozenset[str] = frozenset({"M1", "M5"})
# OANDA v20's candles endpoint hard-caps ``count`` at 5,000 rows. This is an
# upstream API boundary rather than a tuned market parameter; change it only
# if OANDA changes the endpoint contract.
OANDA_CANDLE_COUNT_MAX = 5000
# Keep serialized quarantine evidence bounded inside each of the seven
# pair-chart timeframes. Counts still cover the complete requested packet.
TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT = 8
TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER = "CHRONOLOGICAL_ASC_NULLS_FIRST"
TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION = "LATEST_BOUNDED_WINDOW"
# ``compute_indicators`` deliberately marks a panel invalid below 30 bars,
# and ``chart_reader`` publishes the same 30-bar recent evidence window.  A
# quarantined packet therefore needs both 30 clean observations in total and
# 30 clean observations after the most recent quarantined row before it may
# drive a forecast.
TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT = 30
_OANDA_MBA_UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z"
)
_FIXED_UTC_CADENCE_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
}


@dataclass(frozen=True)
class Candle:
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    complete: bool = True


@dataclass(frozen=True)
class TechnicalCandleBatch:
    """Clean MID candles plus bounded bid/ask integrity evidence."""

    candles: tuple[Candle, ...]
    integrity: dict[str, Any]


def fetch_candles(
    pair: str,
    granularity: str,
    *,
    count: int = 200,
    price: str = "M",
    client: OandaReadOnlyClient | None = None,
) -> tuple[Candle, ...]:
    """Fetch up to `count` recent candles for a pair-granularity.

    `price="M"` returns mid candles (default). Use `"BA"` for bid/ask if needed
    by callers — the parser pulls whichever block is present, preferring mid.
    """

    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"unsupported granularity {granularity!r}; expected one of {sorted(SUPPORTED_GRANULARITIES)}")
    client = client or OandaReadOnlyClient()
    payload = client.get_json(
        f"/v3/instruments/{pair}/candles",
        {"granularity": granularity, "count": str(int(count)), "price": price},
    )
    return _candles_from_payload(payload)


def fetch_candles_between(
    pair: str,
    granularity: str,
    *,
    time_from: datetime,
    time_to: datetime,
    price: str = "M",
    client: OandaReadOnlyClient | None = None,
) -> tuple[Candle, ...]:
    """Fetch complete candles inside an explicit UTC time window.

    This is read-only historical evidence plumbing. Callers that need long
    windows are responsible for chunking the request; OANDA caps candle result
    sizes, and hiding pagination here would make long evidence jobs opaque.
    """

    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"unsupported granularity {granularity!r}; expected one of {sorted(SUPPORTED_GRANULARITIES)}")
    client = client or OandaReadOnlyClient()
    payload = client.get_json(
        f"/v3/instruments/{pair}/candles",
        {
            "granularity": granularity,
            "from": _format_oanda_time(time_from),
            "to": _format_oanda_time(time_to),
            "price": price,
            "includeFirst": "true",
        },
    )
    return _candles_from_payload(payload)


def fetch_candles_via_client(
    client: OandaReadOnlyClient,
    pair: str,
    granularity: str,
    *,
    count: int = 200,
    price: str = "M",
) -> tuple[Candle, ...]:
    """Same as `fetch_candles` but with the client passed positionally for tests."""

    return fetch_candles(pair, granularity, count=count, price=price, client=client)


def fetch_technical_candles_via_client(
    client: OandaReadOnlyClient,
    pair: str,
    granularity: str,
    *,
    count: int = 200,
) -> TechnicalCandleBatch:
    """Fetch one OANDA MBA packet and quarantine spread-distorted MID bars.

    The indicator series remains MID-based. M1/M5 endpoint BID/ASK widths are
    checked against the observed maximum in the pinned spread-calibration
    cohort. This is deliberately separate from ``RiskEngine``'s current-quote
    cap, whose P95-derived value is an execution-cost boundary rather than a
    historical-data corruption threshold. Higher-TF MBA remains strict
    provenance, but its independently aggregated window extrema do not
    masquerade as an executable spread. One MBA request avoids cross-request
    timestamp drift and never substitutes BID for MID.
    """

    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"unsupported granularity {granularity!r}; expected one of {sorted(SUPPORTED_GRANULARITIES)}")
    if count.__class__ is not int or not 1 <= count <= OANDA_CANDLE_COUNT_MAX:
        raise ValueError(
            f"{TECHNICAL_CANDLE_PROVENANCE_INVALID}: count must be an exact integer in "
            f"[1, {OANDA_CANDLE_COUNT_MAX}]"
        )
    from quant_rabbit.instruments import (
        NORMAL_SPREAD_PIPS,
        OANDA_SPREAD_CALIBRATION_V1,
        instrument_pip_factor,
    )
    from quant_rabbit.risk import RiskPolicy

    pair_key = pair.upper()
    normal_spread_pips = NORMAL_SPREAD_PIPS.get(pair_key)
    spread_calibration = OANDA_SPREAD_CALIBRATION_V1.pairs.get(pair_key)
    max_spread_multiple = RiskPolicy().max_spread_multiple
    if (
        normal_spread_pips is None
        or spread_calibration is None
        or not math.isfinite(float(normal_spread_pips))
        or float(normal_spread_pips) <= 0.0
        or not math.isfinite(float(spread_calibration.max_pips))
        or float(spread_calibration.max_pips) <= 0.0
        or not math.isfinite(float(max_spread_multiple))
        or float(max_spread_multiple) <= 0.0
    ):
        raise ValueError(f"{TECHNICAL_CANDLE_PROVENANCE_INVALID}: canonical spread policy missing for {pair_key}")
    payload = client.get_json(
        f"/v3/instruments/{pair_key}/candles",
        {"granularity": granularity, "count": str(int(count)), "price": "MBA"},
    )
    return _technical_candles_from_payload(
        payload,
        pair=pair_key,
        granularity=granularity,
        requested_count=int(count),
        pip_factor=instrument_pip_factor(pair_key),
        normal_spread_pips=float(normal_spread_pips),
        max_spread_multiple=float(max_spread_multiple),
        spread_anomaly_cap_pips=float(spread_calibration.max_pips),
        spread_calibration_sha256=OANDA_SPREAD_CALIBRATION_V1.calibration_sha256,
    )


def _technical_candles_from_payload(
    payload: object,
    *,
    pair: str,
    granularity: str,
    requested_count: int,
    pip_factor: int,
    normal_spread_pips: float,
    max_spread_multiple: float,
    spread_anomaly_cap_pips: float,
    spread_calibration_sha256: str,
) -> TechnicalCandleBatch:
    """Parse strict OANDA MBA provenance and return only uncontaminated MID."""

    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(
            f"{TECHNICAL_CANDLE_PROVENANCE_INVALID}: unsupported granularity {granularity!r}"
        )
    if pip_factor not in {100, 10000}:
        raise ValueError(f"{TECHNICAL_CANDLE_PROVENANCE_INVALID}: unsupported pip factor {pip_factor!r}")
    if (
        requested_count.__class__ is not int
        or requested_count <= 0
        or requested_count > OANDA_CANDLE_COUNT_MAX
    ):
        raise ValueError(
            f"{TECHNICAL_CANDLE_PROVENANCE_INVALID}: requested_count must be an exact positive OANDA count"
        )
    indicator_warmup_min_clean_count = (
        TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT
        if granularity in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES
        else 1
    )
    price_decimal_places = 3 if pip_factor == 100 else 5
    if (
        not math.isfinite(spread_anomaly_cap_pips)
        or spread_anomaly_cap_pips <= 0.0
        or not isinstance(spread_calibration_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", spread_calibration_sha256)
    ):
        raise ValueError(
            f"{TECHNICAL_CANDLE_PROVENANCE_INVALID}: spread anomaly calibration invalid"
        )
    cap_pips = spread_anomaly_cap_pips
    raw_payload_instrument = payload.get("instrument") if payload.__class__ is dict else None
    raw_payload_granularity = payload.get("granularity") if payload.__class__ is dict else None
    entries = payload.get("candles") if payload.__class__ is dict else None
    if (
        raw_payload_instrument.__class__ is not str
        or raw_payload_instrument != pair
        or raw_payload_granularity.__class__ is not str
        or raw_payload_granularity != granularity
    ):
        return TechnicalCandleBatch(
            candles=(),
            integrity=_technical_candle_integrity(
                pair=pair,
                granularity=granularity,
                payload_instrument=_bounded_payload_identity(raw_payload_instrument),
                payload_granularity=_bounded_payload_identity(raw_payload_granularity),
                requested_count=requested_count,
                raw_entry_count=len(entries) if isinstance(entries, list) else 0,
                complete_entry_count=0,
                clean_count=0,
                recent_clean_tail_count=0,
                indicator_warmup_min_clean_count=indicator_warmup_min_clean_count,
                contaminated_count=0,
                malformed_count=1,
                normal_spread_pips=normal_spread_pips,
                max_spread_multiple=max_spread_multiple,
                cap_pips=cap_pips,
                spread_calibration_sha256=spread_calibration_sha256,
                recent_tail_state="PROVENANCE_INVALID",
                latest_complete_timestamp_utc=None,
                latest_clean_timestamp_utc=None,
                quarantine_details=({
                    "timestamp_utc": None,
                    "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    "reason": "payload instrument/granularity missing or mismatched",
                },),
            ),
        )
    if not isinstance(entries, list):
        return TechnicalCandleBatch(
            candles=(),
            integrity=_technical_candle_integrity(
                pair=pair,
                granularity=granularity,
                payload_instrument=raw_payload_instrument,
                payload_granularity=raw_payload_granularity,
                requested_count=requested_count,
                raw_entry_count=0,
                complete_entry_count=0,
                clean_count=0,
                recent_clean_tail_count=0,
                indicator_warmup_min_clean_count=indicator_warmup_min_clean_count,
                contaminated_count=0,
                malformed_count=1,
                normal_spread_pips=normal_spread_pips,
                max_spread_multiple=max_spread_multiple,
                cap_pips=cap_pips,
                spread_calibration_sha256=spread_calibration_sha256,
                recent_tail_state="PROVENANCE_INVALID",
                latest_complete_timestamp_utc=None,
                latest_clean_timestamp_utc=None,
                quarantine_details=({
                    "timestamp_utc": None,
                    "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    "reason": "candles array missing or malformed",
                },),
            ),
        )

    clean: list[Candle] = []
    recent_clean_tail_count = 0
    complete_entry_count = 0
    contaminated_count = 0
    packet_count_mismatch = len(entries) != requested_count
    malformed_count = 1 if packet_count_mismatch else 0
    quarantine_details: list[dict[str, Any]] = []
    if packet_count_mismatch:
        quarantine_details.append({
            "timestamp_utc": None,
            "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
            "reason": (
                "OANDA candle count does not match the exact requested count "
                f"({len(entries)} != {requested_count})"
            ),
        })
    latest_complete_timestamp: datetime | None = None
    latest_clean_timestamp: datetime | None = None
    recent_tail_state: str | None = None
    prior_timestamp: datetime | None = None

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            recent_clean_tail_count = 0
            malformed_count += 1
            recent_tail_state = "PROVENANCE_INVALID"
            quarantine_details.append({
                "timestamp_utc": None,
                "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                "reason": f"candles[{index}] is not an object",
            })
            continue
        complete = entry.get("complete")
        if complete is False:
            if index != len(entries) - 1:
                recent_clean_tail_count = 0
                malformed_count += 1
                recent_tail_state = "PROVENANCE_INVALID"
                quarantine_details.append({
                    "timestamp_utc": _bounded_evidence_text(entry.get("time")),
                    "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    "reason": f"incomplete candles[{index}] must be the packet tail",
                })
            continue
        if complete is not True:
            recent_clean_tail_count = 0
            malformed_count += 1
            recent_tail_state = "PROVENANCE_INVALID"
            quarantine_details.append({
                "timestamp_utc": _bounded_evidence_text(entry.get("time")),
                "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                "reason": f"candles[{index}].complete must be an exact boolean",
            })
            continue

        complete_entry_count += 1
        timestamp = _parse_oanda_mba_time(entry.get("time"))
        mid = _strict_ohlc_block(entry.get("mid"), decimal_places=price_decimal_places)
        bid = _strict_ohlc_block(entry.get("bid"), decimal_places=price_decimal_places)
        ask = _strict_ohlc_block(entry.get("ask"), decimal_places=price_decimal_places)
        volume = entry.get("volume")
        provenance_error: str | None = None
        if timestamp is None:
            provenance_error = "timestamp missing or malformed"
        elif prior_timestamp is not None and timestamp <= prior_timestamp:
            provenance_error = "timestamps are duplicate or out of order"
        elif not _technical_candle_cadence_valid(
            granularity=granularity,
            timestamp=timestamp,
            prior_timestamp=prior_timestamp,
            raw_timestamp=entry.get("time"),
        ):
            provenance_error = (
                "timestamp is not aligned to the requested granularity cadence"
            )
        elif mid is None or bid is None or ask is None:
            provenance_error = "complete candle requires valid mid, bid, and ask OHLC"
        elif not _mba_ohlc_blocks_consistent(mid=mid, bid=bid, ask=ask, pip_factor=pip_factor):
            provenance_error = (
                "mid OHLC must stay inside bid/ask and O/C must match the "
                "endpoint midpoint within OANDA precision"
            )
        elif not isinstance(volume, int) or isinstance(volume, bool) or volume < 0:
            provenance_error = "volume must be an exact non-negative integer"

        if timestamp is not None:
            prior_timestamp = timestamp
            latest_complete_timestamp = timestamp
        if provenance_error is not None:
            recent_clean_tail_count = 0
            malformed_count += 1
            recent_tail_state = "PROVENANCE_INVALID"
            quarantine_details.append({
                "timestamp_utc": timestamp.isoformat() if timestamp else _bounded_evidence_text(entry.get("time")),
                "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                "reason": provenance_error,
            })
            continue

        assert timestamp is not None and mid is not None and bid is not None and ask is not None
        # OANDA's H/L values are independent extrema over the candle window,
        # not simultaneous executable quotes.  Only O/C are point-in-time MBA
        # endpoints whose BID/ASK distance can be compared as spread.
        spreads = tuple((ask[key] - bid[key]) * pip_factor for key in ("o", "c"))
        if any(not math.isfinite(value) or value < 0.0 for value in spreads):
            recent_clean_tail_count = 0
            malformed_count += 1
            recent_tail_state = "PROVENANCE_INVALID"
            quarantine_details.append({
                "timestamp_utc": timestamp.isoformat(),
                "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                "reason": "ask OHLC must not be below bid OHLC",
            })
            continue
        max_spread_pips = max(spreads)
        # OANDA prices arrive as decimal strings but Python arithmetic is
        # binary floating point. Treat only machine-noise equality as the cap;
        # any economically measurable excess remains quarantined.
        if (
            granularity in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES
            and max_spread_pips > cap_pips
            and not math.isclose(max_spread_pips, cap_pips, rel_tol=0.0, abs_tol=1e-9)
        ):
            recent_clean_tail_count = 0
            contaminated_count += 1
            recent_tail_state = "SPREAD_CONTAMINATED"
            quarantine_details.append({
                "timestamp_utc": timestamp.isoformat(),
                "code": TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
                "max_spread_pips": round(max_spread_pips, 6),
                "spread_cap_pips": round(cap_pips, 6),
            })
            continue

        clean.append(Candle(
            timestamp_utc=timestamp,
            open=mid["o"],
            high=mid["h"],
            low=mid["l"],
            close=mid["c"],
            volume=volume,
            complete=True,
        ))
        # A genuine no-tick gap has already passed cadence validation above,
        # so it remains part of one clean observation run.  Only an observed
        # contaminated/malformed row resets this latest-side coverage.
        recent_clean_tail_count += 1
        latest_clean_timestamp = timestamp
        recent_tail_state = "CLEAN"

    if complete_entry_count == 0 and malformed_count == 0:
        malformed_count = 1
        recent_tail_state = "PROVENANCE_INVALID"
        quarantine_details.append({
            "timestamp_utc": None,
            "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
            "reason": "no complete candle available",
        })
    if packet_count_mismatch:
        # Missing rows mean the newest requested coverage itself is unknown;
        # a clean-looking last returned row cannot repair that provenance gap.
        recent_tail_state = "PROVENANCE_INVALID"

    return TechnicalCandleBatch(
        candles=tuple(clean),
        integrity=_technical_candle_integrity(
            pair=pair,
            granularity=granularity,
            payload_instrument=raw_payload_instrument,
            payload_granularity=raw_payload_granularity,
            requested_count=requested_count,
            raw_entry_count=len(entries),
            complete_entry_count=complete_entry_count,
            clean_count=len(clean),
            recent_clean_tail_count=recent_clean_tail_count,
            indicator_warmup_min_clean_count=indicator_warmup_min_clean_count,
            contaminated_count=contaminated_count,
            malformed_count=malformed_count,
            normal_spread_pips=normal_spread_pips,
            max_spread_multiple=max_spread_multiple,
            cap_pips=cap_pips,
            spread_calibration_sha256=spread_calibration_sha256,
            recent_tail_state=recent_tail_state,
            latest_complete_timestamp_utc=latest_complete_timestamp,
            latest_clean_timestamp_utc=latest_clean_timestamp,
            quarantine_details=tuple(quarantine_details),
        ),
    )


def _technical_candle_cadence_valid(
    *,
    granularity: str,
    timestamp: datetime,
    prior_timestamp: datetime | None,
    raw_timestamp: object = None,
) -> bool:
    """Validate OANDA boundary cadence while allowing genuine no-tick gaps.

    Fixed UTC intraday bars must land on their exact epoch boundary and any
    gap must be an integer number of bars. H4/D use OANDA's New York-aligned
    daily boundary, which can shift by one hour at DST, so those frames retain
    whole-hour alignment with a bounded minimum duration instead of assuming a
    fixed UTC phase. This rejects time-compressed synthetic series without
    inventing flat candles for real no-tick intervals.
    """

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return False
    timestamp = timestamp.astimezone(timezone.utc)
    if timestamp.microsecond != 0:
        return False
    if raw_timestamp.__class__ is str and "." in raw_timestamp:
        fraction = raw_timestamp.split(".", 1)[1][:-1]
        if not fraction or any(char != "0" for char in fraction):
            return False

    fixed_seconds = _FIXED_UTC_CADENCE_SECONDS.get(granularity)
    if fixed_seconds is not None:
        epoch_seconds = int(timestamp.timestamp())
        if epoch_seconds % fixed_seconds != 0:
            return False
        if prior_timestamp is None:
            return True
        delta_seconds = int((timestamp - prior_timestamp).total_seconds())
        return (
            delta_seconds >= fixed_seconds
            and delta_seconds % fixed_seconds == 0
        )

    if granularity not in {"H4", "D"}:
        return False
    if timestamp.minute != 0 or timestamp.second != 0:
        return False
    if prior_timestamp is None:
        return True
    delta_seconds_float = (timestamp - prior_timestamp).total_seconds()
    if not delta_seconds_float.is_integer():
        return False
    delta_seconds = int(delta_seconds_float)
    minimum_seconds = 3 * 60 * 60 if granularity == "H4" else 23 * 60 * 60
    return delta_seconds >= minimum_seconds and delta_seconds % (60 * 60) == 0


def _strict_ohlc_block(
    value: object,
    *,
    decimal_places: int,
) -> dict[str, float] | None:
    """Return a finite, internally consistent OANDA OHLC block."""

    if not isinstance(value, dict):
        return None
    parsed: dict[str, float] = {}
    decimal_pattern = re.compile(rf"(?:0|[1-9][0-9]*)\.[0-9]{{{decimal_places}}}")
    for key in ("o", "h", "l", "c"):
        raw = value.get(key)
        if raw.__class__ is not str or raw != raw.strip() or decimal_pattern.fullmatch(raw) is None:
            return None
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0.0:
            return None
        parsed[key] = number
    if not (parsed["l"] <= min(parsed["o"], parsed["c"]) <= max(parsed["o"], parsed["c"]) <= parsed["h"]):
        return None
    return parsed


def _mba_ohlc_blocks_consistent(
    *,
    mid: dict[str, float],
    bid: dict[str, float],
    ask: dict[str, float],
    pip_factor: int,
) -> bool:
    # OANDA quotes FX one decimal place finer than one pip. Rounding an exact
    # bid/ask midpoint to that display precision can move it by at most 0.05pip.
    midpoint_tolerance_pips = 0.050000001
    for key in ("o", "h", "l", "c"):
        if not bid[key] <= mid[key] <= ask[key]:
            return False
    # OANDA computes each component's extrema independently.  MID high/low
    # therefore need not equal the arithmetic midpoint of BID/ASK high/low,
    # even though they must remain inside the executable quote envelope.
    # Open/close are point-in-time observations and retain the tighter
    # midpoint provenance check.
    for key in ("o", "c"):
        midpoint = (bid[key] + ask[key]) / 2.0
        if abs(mid[key] - midpoint) * pip_factor > midpoint_tolerance_pips:
            return False
    return True


def _bounded_evidence_text(value: object, *, limit: int = 256) -> str | None:
    text = str(value or "")
    return text[:limit] or None


def _bounded_payload_identity(value: object, *, limit: int = 64) -> str | None:
    return value[:limit] if value.__class__ is str and value else None


def _quarantine_detail_timestamp(detail: dict[str, Any]) -> datetime | None:
    value = detail.get("timestamp_utc")
    if value.__class__ is not str or not value:
        return None
    text = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _canonical_quarantine_details(
    details: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Sort quarantine evidence chronologically with unknown clocks first."""

    minimum = datetime.min.replace(tzinfo=timezone.utc)

    def _key(detail: dict[str, Any]) -> tuple[object, ...]:
        timestamp = _quarantine_detail_timestamp(detail)
        tie_breaker = tuple(
            (key, repr(detail[key])) for key in sorted(detail)
        )
        return (
            0 if timestamp is None else 1,
            timestamp or minimum,
            tie_breaker,
        )

    return tuple(sorted(details, key=_key))


def _quarantine_code_counts(
    details: tuple[dict[str, Any], ...],
) -> dict[str, int]:
    """Count the two producer-owned quarantine classes exactly."""

    return {
        TECHNICAL_CANDLE_SPREAD_CONTAMINATED: sum(
            detail.get("code") == TECHNICAL_CANDLE_SPREAD_CONTAMINATED
            for detail in details
        ),
        TECHNICAL_CANDLE_PROVENANCE_INVALID: sum(
            detail.get("code") == TECHNICAL_CANDLE_PROVENANCE_INVALID
            for detail in details
        ),
    }


def _technical_candle_integrity(
    *,
    pair: str,
    granularity: str,
    payload_instrument: str | None,
    payload_granularity: str | None,
    requested_count: int,
    raw_entry_count: int,
    complete_entry_count: int,
    clean_count: int,
    recent_clean_tail_count: int,
    indicator_warmup_min_clean_count: int,
    contaminated_count: int,
    malformed_count: int,
    normal_spread_pips: float,
    max_spread_multiple: float,
    cap_pips: float,
    spread_calibration_sha256: str,
    recent_tail_state: str | None,
    latest_complete_timestamp_utc: datetime | None,
    latest_clean_timestamp_utc: datetime | None,
    quarantine_details: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Build bounded, machine-readable provenance for one timeframe."""

    codes: list[str] = []
    blocking_codes: list[str] = []
    if contaminated_count > 0:
        codes.append(TECHNICAL_CANDLE_SPREAD_CONTAMINATED)
    warmup_complete = clean_count >= indicator_warmup_min_clean_count
    recent_clean_coverage_complete = (
        recent_clean_tail_count >= indicator_warmup_min_clean_count
    )
    clean_coverage_blocked = (
        recent_tail_state == "CLEAN"
        and (not warmup_complete or not recent_clean_coverage_complete)
    )
    if (
        malformed_count > 0
        or complete_entry_count <= 0
        or clean_count <= 0
        or clean_coverage_blocked
    ):
        codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)
    if recent_tail_state == "SPREAD_CONTAMINATED":
        blocking_codes.append(TECHNICAL_CANDLE_SPREAD_CONTAMINATED)
    if (
        malformed_count > 0
        or recent_tail_state == "PROVENANCE_INVALID"
        or complete_entry_count <= 0
        or clean_count <= 0
        or clean_coverage_blocked
    ):
        blocking_codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)

    codes = list(dict.fromkeys(codes))
    blocking_codes = list(dict.fromkeys(blocking_codes))
    forecast_blocking = bool(blocking_codes)
    if forecast_blocking:
        evaluation_status = "BLOCKED"
    elif contaminated_count > 0:
        evaluation_status = "DEGRADED"
    else:
        evaluation_status = "PASS"

    canonical_details = _canonical_quarantine_details(quarantine_details)
    published_details = canonical_details[-TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT:]
    omitted_details = canonical_details[:-TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT]
    quarantine_total = len(canonical_details)
    quarantine_window_start = max(
        0,
        quarantine_total - TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
    )
    latest_quarantine_timestamp = next(
        (
            timestamp
            for timestamp in reversed(tuple(
                _quarantine_detail_timestamp(detail)
                for detail in canonical_details
            ))
            if timestamp is not None
        ),
        None,
    )
    total_code_counts = _quarantine_code_counts(canonical_details)
    published_code_counts = _quarantine_code_counts(published_details)
    omitted_code_counts = _quarantine_code_counts(omitted_details)
    total_timestamped_count = sum(
        _quarantine_detail_timestamp(detail) is not None
        for detail in canonical_details
    )
    published_timestamped_count = sum(
        _quarantine_detail_timestamp(detail) is not None
        for detail in published_details
    )
    return {
        "schema": TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
        "pair": pair,
        "granularity": granularity,
        "payload_instrument": payload_instrument,
        "payload_granularity": payload_granularity,
        "source": "OANDA_MBA",
        "requested_price": "MBA",
        "spread_evaluation_mode": _spread_evaluation_mode(granularity),
        "evaluation_status": evaluation_status,
        "policy_source": "OANDA_SPREAD_CALIBRATION_V1.pairs.max_pips",
        "spread_calibration_sha256": spread_calibration_sha256,
        "normal_spread_pips": round(normal_spread_pips, 6),
        "max_spread_multiple": round(max_spread_multiple, 6),
        "execution_spread_cap_pips": round(
            normal_spread_pips * max_spread_multiple,
            6,
        ),
        "spread_cap_pips": round(cap_pips, 6),
        "requested_count": requested_count,
        "raw_entry_count": raw_entry_count,
        "coverage_complete": raw_entry_count == requested_count,
        "complete_entry_count": complete_entry_count,
        "clean_count": clean_count,
        "indicator_warmup_min_clean_count": indicator_warmup_min_clean_count,
        "recent_clean_tail_count": recent_clean_tail_count,
        "indicator_warmup_complete": warmup_complete,
        "recent_clean_coverage_complete": recent_clean_coverage_complete,
        "contaminated_count": contaminated_count,
        "malformed_count": malformed_count,
        "quarantined_count": contaminated_count + malformed_count,
        "recent_tail_state": recent_tail_state or "PROVENANCE_INVALID",
        "recent_tail_contaminated": recent_tail_state == "SPREAD_CONTAMINATED",
        "recent_tail_invalid": recent_tail_state == "PROVENANCE_INVALID",
        "provenance_complete": malformed_count == 0 and complete_entry_count > 0,
        "forecast_blocking": forecast_blocking,
        "codes": codes,
        "blocking_codes": blocking_codes,
        "latest_complete_timestamp_utc": latest_complete_timestamp_utc.isoformat() if latest_complete_timestamp_utc else None,
        "latest_clean_timestamp_utc": latest_clean_timestamp_utc.isoformat() if latest_clean_timestamp_utc else None,
        "quarantine_details": list(published_details),
        "quarantine_details_truncated": quarantine_window_start,
        "quarantine_details_window": {
            "selection": TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION,
            "order": TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER,
            "limit": TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
            "start_index": quarantine_window_start,
            "end_index_exclusive": quarantine_total,
            "total_count": quarantine_total,
            "total_code_counts": total_code_counts,
            "published_code_counts": published_code_counts,
            "omitted_code_counts": omitted_code_counts,
            "total_timestamped_count": total_timestamped_count,
            "published_timestamped_count": published_timestamped_count,
            "omitted_timestamped_count": (
                total_timestamped_count - published_timestamped_count
            ),
            "latest_timestamp_utc": (
                latest_quarantine_timestamp.isoformat()
                if latest_quarantine_timestamp is not None
                else None
            ),
        },
    }


def _spread_evaluation_mode(granularity: str) -> str:
    return (
        TECHNICAL_CANDLE_SPREAD_EXECUTION_MODE
        if granularity in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES
        else TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE
    )


def _candles_from_payload(payload: dict) -> tuple[Candle, ...]:
    candles: list[Candle] = []
    for entry in payload.get("candles") or []:
        # Strategy indicators and structure must never repaint from OANDA's
        # still-forming tail candle. Live bid/ask quotes are consumed through
        # broker snapshots, so excluding an incomplete bar does not make the
        # guardian blind to price displacement; it keeps the technical state
        # anchored to closed evidence as downstream pattern code expects.
        if entry.get("complete") is False:
            continue
        block = entry.get("mid") or entry.get("ask") or entry.get("bid") or {}
        try:
            timestamp = _parse_oanda_time(entry.get("time"))
            o = float(block.get("o"))
            h = float(block.get("h"))
            low_value = float(block.get("l"))
            c = float(block.get("c"))
        except (TypeError, ValueError):
            continue
        if timestamp is None:
            continue
        candles.append(
            Candle(
                timestamp_utc=timestamp,
                open=o,
                high=h,
                low=low_value,
                close=c,
                volume=int(entry.get("volume") or 0),
                complete=bool(entry.get("complete", True)),
            )
        )
    return tuple(candles)


def _parse_oanda_time(value: object) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        core = text[:-1]
        if "." in core:
            head, frac = core.split(".", 1)
            text = f"{head}.{frac[:6]}+00:00"
        else:
            text = f"{core}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _parse_oanda_mba_time(value: object) -> datetime | None:
    if value.__class__ is not str or _OANDA_MBA_UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        return None
    return _parse_oanda_time(value)


def _format_oanda_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def closes(candles: Sequence[Candle]) -> tuple[float, ...]:
    return tuple(c.close for c in candles)


def highs(candles: Sequence[Candle]) -> tuple[float, ...]:
    return tuple(c.high for c in candles)


def lows(candles: Sequence[Candle]) -> tuple[float, ...]:
    return tuple(c.low for c in candles)


def volumes(candles: Sequence[Candle]) -> tuple[int, ...]:
    return tuple(c.volume for c in candles)
