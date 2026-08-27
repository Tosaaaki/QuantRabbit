"""Independent scalar reference calculator for JPY accounting fixtures.

This file intentionally does not import ``jpy_accounting_v2``.  It duplicates
the small executable-conversion and linear-P/L equations so tests can compare
the runtime implementation with an independent calculation path.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence


class ReferenceError(RuntimeError):
    pass


UTC_NANOSECOND_RE = re.compile(
    r"^(?P<second>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)


def parse_utc_nanoseconds(value: str) -> int:
    """Parse canonical UTC RFC3339 without truncating 7-9 digit fractions."""
    match = UTC_NANOSECOND_RE.fullmatch(value)
    if match is None:
        raise ReferenceError(f"invalid UTC timestamp: {value}")
    try:
        second = datetime.strptime(
            match.group("second"), "%Y-%m-%dT%H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise ReferenceError(f"invalid UTC timestamp: {value}") from error
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return calendar.timegm(second.utctimetuple()) * 1_000_000_000 + int(fraction or "0")


def currencies(pair: str) -> tuple[str, str]:
    fields = pair.split("_")
    if len(fields) != 2 or any(len(field) != 3 for field in fields):
        raise ReferenceError(f"invalid pair: {pair}")
    return fields[0], fields[1]


def sell_currency(
    amount: float,
    source: str,
    target: str,
    pair: str,
    bid: float,
    ask: float,
) -> float:
    base, quote = currencies(pair)
    if source == base and target == quote:
        return amount * bid
    if source == quote and target == base:
        return amount / ask
    raise ReferenceError(f"pair {pair} cannot convert {source}->{target}")


def asset_rate_to_jpy(
    currency: str,
    path: Sequence[tuple[str, str, str]],
    quotes: Mapping[str, tuple[float, float]],
) -> float:
    if currency == "JPY":
        return 1.0
    value = 1.0
    current = currency
    for pair, source, target in path:
        if source != current:
            raise ReferenceError("disconnected asset path")
        bid, ask = quotes[pair]
        value = sell_currency(value, source, target, pair, bid, ask)
        current = target
    if current != "JPY":
        raise ReferenceError("asset path does not end in JPY")
    return value


def liability_rate_to_jpy(
    currency: str,
    path: Sequence[tuple[str, str, str]],
    quotes: Mapping[str, tuple[float, float]],
) -> float:
    if currency == "JPY":
        return 1.0
    source_per_jpy = 1.0
    current = "JPY"
    for pair, source, target in reversed(path):
        if target != current:
            raise ReferenceError("disconnected liability path")
        bid, ask = quotes[pair]
        source_per_jpy = sell_currency(
            source_per_jpy, target, source, pair, bid, ask
        )
        current = source
    if current != currency or source_per_jpy <= 0.0:
        raise ReferenceError("liability path does not end in source currency")
    return 1.0 / source_per_jpy


def convert_to_jpy(
    amount: float,
    currency: str,
    path: Sequence[tuple[str, str, str]],
    quotes: Mapping[str, tuple[float, float]],
) -> float:
    if not math.isfinite(amount):
        raise ReferenceError("amount must be finite")
    if amount >= 0.0:
        return amount * asset_rate_to_jpy(currency, path, quotes)
    return amount * liability_rate_to_jpy(currency, path, quotes)


def episode(
    *,
    pair: str,
    direction: int,
    notional_jpy: float,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
    entry_conversion_quotes: Mapping[str, tuple[float, float]],
    exit_conversion_quotes: Mapping[str, tuple[float, float]],
    quote_to_jpy_path: Sequence[tuple[str, str, str]],
    entry_time: str,
    exit_time: str,
    slippage_pips: float,
    commission_bps_per_side: float,
    financing_bps_per_day: float,
    raw_pair_mid: bool,
) -> dict[str, float]:
    if direction not in (-1, 1):
        raise ReferenceError("direction must be -1 or +1")
    quote_currency = currencies(pair)[1]
    pip = 0.01 if pair.endswith("_JPY") else 0.0001
    sizing_price = entry_ask if direction > 0 else entry_bid
    opening_quote_cashflow = -direction * sizing_price
    jpy_per_unit = abs(convert_to_jpy(
        opening_quote_cashflow, quote_currency,
        quote_to_jpy_path, entry_conversion_quotes,
    ))
    units = notional_jpy / jpy_per_unit

    entry_mid = (entry_bid + entry_ask) / 2.0
    exit_mid = (exit_bid + exit_ask) / 2.0
    raw_quote_pnl = direction * units * (exit_mid - entry_mid)
    gross_jpy = convert_to_jpy(
        raw_quote_pnl, quote_currency, quote_to_jpy_path, exit_conversion_quotes
    )

    if raw_pair_mid:
        entry_price, exit_price = entry_mid, exit_mid
    else:
        slip = slippage_pips * pip
        entry_price = entry_ask + slip if direction > 0 else entry_bid - slip
        exit_price = exit_bid - slip if direction > 0 else exit_ask + slip
    pair_quote_pnl = direction * units * (exit_price - entry_price)
    pair_jpy = convert_to_jpy(
        pair_quote_pnl, quote_currency, quote_to_jpy_path, exit_conversion_quotes
    )
    elapsed_nanoseconds = (
        parse_utc_nanoseconds(exit_time) - parse_utc_nanoseconds(entry_time)
    )
    if elapsed_nanoseconds < 0:
        raise ReferenceError("exit time precedes entry time")
    seconds = elapsed_nanoseconds / 1_000_000_000.0
    days = seconds / 86_400.0
    entry_commission_quote = units * entry_price * commission_bps_per_side * 1e-4
    exit_commission_quote = units * exit_price * commission_bps_per_side * 1e-4
    financing_quote = units * entry_price * financing_bps_per_day * 1e-4 * days
    entry_commission_jpy = -convert_to_jpy(
        -entry_commission_quote, quote_currency,
        quote_to_jpy_path, entry_conversion_quotes,
    )
    exit_commission_jpy = -convert_to_jpy(
        -exit_commission_quote, quote_currency,
        quote_to_jpy_path, exit_conversion_quotes,
    )
    financing_jpy = -convert_to_jpy(
        -financing_quote, quote_currency,
        quote_to_jpy_path, exit_conversion_quotes,
    )
    net_jpy = pair_jpy - entry_commission_jpy - exit_commission_jpy - financing_jpy
    return {
        "units": units,
        "raw_quote_pnl": raw_quote_pnl,
        "gross_jpy": gross_jpy,
        "executable_pair_quote_pnl": pair_quote_pnl,
        "executable_pair_jpy": pair_jpy,
        "commission_cost_jpy": entry_commission_jpy + exit_commission_jpy,
        "financing_cost_jpy": financing_jpy,
        "net_jpy": net_jpy,
        "elapsed_seconds": seconds,
    }
