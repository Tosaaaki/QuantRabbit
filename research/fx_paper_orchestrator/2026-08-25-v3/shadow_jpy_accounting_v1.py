"""Standalone paper JPY accounting primitive for the offline shadow core.

This module intentionally has no strategy/runtime import chain.  It contains
only the small, fixed accounting surface required by forward_shadow_core_v1.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


_PAIR = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_STAMP = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?Z$"
)
MAX_STALENESS_NS = 300 * 1_000_000_000


class AccountingError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _embedded(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _ns(value: str) -> int:
    match = _STAMP.fullmatch(value)
    if match is None:
        raise AccountingError(f"invalid UTC timestamp: {value}")
    year, month, day, hour, minute, second = map(int, match.groups()[:6])
    fraction = (match.group(7) or "").ljust(9, "0")
    try:
        base = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError as error:
        raise AccountingError(f"invalid UTC timestamp: {value}") from error
    return int(base.timestamp()) * 1_000_000_000 + int(fraction or "0")


def pair_currencies(instrument: str) -> tuple[str, str]:
    if _PAIR.fullmatch(instrument) is None:
        raise AccountingError(f"invalid pair: {instrument}")
    base, quote = instrument.split("_", 1)
    return base, quote


def _pip(instrument: str) -> float:
    return 0.01 if instrument.endswith("_JPY") else 0.0001


@dataclass(frozen=True)
class BBO:
    instrument: str
    source_time: str
    bid: float
    ask: float

    def __post_init__(self) -> None:
        pair_currencies(self.instrument)
        _ns(self.source_time)
        if not all(math.isfinite(value) and value > 0 for value in (self.bid, self.ask)):
            raise AccountingError("invalid BBO price")
        if self.ask < self.bid:
            raise AccountingError("inverted BBO")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class CostScenario:
    name: str
    slippage_pips_per_side: float
    commission_bps_per_side: float
    financing_bps_per_day: float
    raw_pair_mid: bool


RAW_SIGNAL = CostScenario("RAW_SIGNAL", 0.0, 0.0, 0.0, True)
EXECUTABLE_BASE = CostScenario("EXECUTABLE_BASE", 0.3, 0.0, 0.5, False)
ADVERSE_STRESS = CostScenario("ADVERSE_STRESS", 0.9, 0.2, 1.5, False)


@dataclass(frozen=True)
class ConversionResult:
    jpy_amount: float
    path: tuple[str, ...]


class ConversionBook:
    def __init__(self, events: Iterable[BBO]) -> None:
        self._events: dict[str, list[BBO]] = {}
        self._times: dict[str, list[int]] = {}
        for event in events:
            values = self._events.setdefault(event.instrument, [])
            times = self._times.setdefault(event.instrument, [])
            stamp = _ns(event.source_time)
            if times and stamp <= times[-1]:
                raise AccountingError("duplicate or reversed conversion chronology")
            values.append(event)
            times.append(stamp)

    def _quote(self, instrument: str, event_time: str) -> BBO:
        target = _ns(event_time)
        times = self._times.get(instrument, [])
        index = bisect.bisect_right(times, target) - 1
        if index < 0:
            raise AccountingError(f"missing causal conversion quote: {instrument}")
        if target - times[index] > MAX_STALENESS_NS:
            raise AccountingError(f"stale conversion quote: {instrument}")
        return self._events[instrument][index]

    def _asset_rate(self, currency: str, event_time: str) -> tuple[float, tuple[str, ...]]:
        if currency == "JPY":
            return 1.0, ()
        usd_jpy = self._quote("USD_JPY", event_time)
        if currency == "USD":
            return usd_jpy.bid, ("USD_JPY",)
        if currency in {"CAD", "CHF"}:
            pair = f"USD_{currency}"
            cross = self._quote(pair, event_time)
            return usd_jpy.bid / cross.ask, (pair, "USD_JPY")
        raise AccountingError(f"missing conversion path: {currency}")

    def _liability_rate(self, currency: str, event_time: str) -> tuple[float, tuple[str, ...]]:
        if currency == "JPY":
            return 1.0, ()
        usd_jpy = self._quote("USD_JPY", event_time)
        if currency == "USD":
            return usd_jpy.ask, ("USD_JPY",)
        if currency in {"CAD", "CHF"}:
            pair = f"USD_{currency}"
            cross = self._quote(pair, event_time)
            return usd_jpy.ask / cross.bid, (pair, "USD_JPY")
        raise AccountingError(f"missing conversion path: {currency}")

    def convert_to_jpy(self, amount: float, currency: str, event_time: str) -> ConversionResult:
        rate, path = (
            self._asset_rate(currency, event_time)
            if amount >= 0 else self._liability_rate(currency, event_time)
        )
        return ConversionResult(amount * rate, path)


@dataclass(frozen=True)
class Position:
    position_id: str
    pair: str
    direction: int
    units: float
    notional_jpy: float
    entry_time: str
    entry_bbo: BBO
    sizing_quote_cashflow_per_unit: float
    sizing_jpy_cashflow_per_unit: float

    @property
    def quote_currency(self) -> str:
        return pair_currencies(self.pair)[1]

    def currency_inventory(self) -> dict[str, float]:
        base, quote = pair_currencies(self.pair)
        return {
            base: self.direction * self.units,
            quote: -self.direction * self.units * self.entry_bbo.mid,
        }


def size_position(
    position_id: str,
    pair: str,
    direction: int,
    notional_jpy: float,
    entry_time: str,
    entry_bbo: BBO,
    conversion_book: ConversionBook,
) -> Position:
    if direction not in (-1, 1) or notional_jpy <= 0 or not math.isfinite(notional_jpy):
        raise AccountingError("invalid position sizing")
    if entry_bbo.instrument != pair or _ns(entry_bbo.source_time) > _ns(entry_time):
        raise AccountingError("entry BBO identity or chronology mismatch")
    quote = pair_currencies(pair)[1]
    cashflow = entry_bbo.ask if direction > 0 else entry_bbo.bid
    per_unit = abs(conversion_book.convert_to_jpy(cashflow, quote, entry_time).jpy_amount)
    if per_unit <= 0:
        raise AccountingError("nonpositive sizing conversion")
    units = notional_jpy / per_unit
    return Position(
        position_id, pair, direction, units, notional_jpy, entry_time,
        entry_bbo, cashflow, per_unit,
    )


def _pair_price(bbo: BBO, direction: int, opening: bool, scenario: CostScenario) -> float:
    if scenario.raw_pair_mid:
        return bbo.mid
    price = bbo.ask if (direction > 0) == opening else bbo.bid
    slip = scenario.slippage_pips_per_side * _pip(bbo.instrument)
    return price + slip if (direction > 0) == opening else price - slip


def evaluate_position(
    position: Position,
    exit_time: str,
    exit_bbo: BBO,
    conversion_book: ConversionBook,
    scenario: CostScenario,
) -> dict[str, Any]:
    if exit_bbo.instrument != position.pair or _ns(exit_bbo.source_time) > _ns(exit_time):
        raise AccountingError("exit BBO identity or chronology mismatch")
    elapsed = (_ns(exit_time) - _ns(position.entry_time)) / 1_000_000_000
    if elapsed < 0:
        raise AccountingError("negative holding time")
    quote = position.quote_currency
    raw_quote = position.direction * position.units * (
        exit_bbo.mid - position.entry_bbo.mid
    )
    gross = conversion_book.convert_to_jpy(raw_quote, quote, exit_time).jpy_amount
    entry_price = _pair_price(position.entry_bbo, position.direction, True, scenario)
    exit_price = _pair_price(exit_bbo, position.direction, False, scenario)
    executable_quote = position.direction * position.units * (exit_price - entry_price)
    executable = conversion_book.convert_to_jpy(executable_quote, quote, exit_time).jpy_amount
    entry_commission = position.units * entry_price * scenario.commission_bps_per_side * 1e-4
    exit_commission = position.units * exit_price * scenario.commission_bps_per_side * 1e-4
    financing = position.units * entry_price * scenario.financing_bps_per_day * 1e-4 * elapsed / 86400
    commission_jpy = -(
        conversion_book.convert_to_jpy(-entry_commission, quote, position.entry_time).jpy_amount
        + conversion_book.convert_to_jpy(-exit_commission, quote, exit_time).jpy_amount
    )
    financing_jpy = -conversion_book.convert_to_jpy(-financing, quote, exit_time).jpy_amount
    net = executable - commission_jpy - financing_jpy
    result = {
        "position_id": position.position_id,
        "scenario": scenario.name,
        "gross_jpy": gross,
        "net_jpy": net,
        "total_realized_cost_jpy": gross - net,
        "financing_cost_jpy": financing_jpy,
        "terminal_inventory_mtm_jpy": 0.0,
    }
    result["evaluation_sha256"] = _embedded(result, "evaluation_sha256")
    return result
