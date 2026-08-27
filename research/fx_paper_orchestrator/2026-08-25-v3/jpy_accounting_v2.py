"""Paper-only JPY account ledger with causal executable currency conversion.

This module is a runtime/accounting migration for future official cycles.  It
does not reinterpret or rewrite any sealed V25-V41 result.  Trade P/L is first
computed linearly in the pair's quote currency.  A positive quote-currency
asset is then sold to JPY, while a negative liability is bought back from JPY,
using only causal BID/ASK quotes at or before the accounting event.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from run_causal_min_spread_representative_v27 import parse_utc_nanoseconds


MAX_CONVERSION_STALENESS_SECONDS = 300
MAX_CONVERSION_STALENESS_NS = MAX_CONVERSION_STALENESS_SECONDS * 1_000_000_000
_PAIR = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")


class AccountingError(RuntimeError):
    """Raised whenever an accounting or causal conversion guard fails."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def embedded_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def pair_currencies(instrument: str) -> tuple[str, str]:
    if _PAIR.fullmatch(instrument) is None:
        raise AccountingError(f"invalid FX instrument: {instrument}")
    return tuple(instrument.split("_", 1))  # type: ignore[return-value]


def pip_size(instrument: str) -> float:
    pair_currencies(instrument)
    return 0.01 if instrument.endswith("_JPY") else 0.0001


@dataclass(frozen=True)
class BBO:
    instrument: str
    source_time: str
    bid: float
    ask: float

    def __post_init__(self) -> None:
        pair_currencies(self.instrument)
        try:
            parse_utc_nanoseconds(self.source_time)
        except ValueError as error:
            raise AccountingError(f"noncanonical BBO timestamp: {self.source_time}") from error
        if not all(math.isfinite(value) and value > 0.0 for value in (self.bid, self.ask)):
            raise AccountingError(f"nonpositive or nonfinite BBO: {self.instrument}")
        if self.ask < self.bid:
            raise AccountingError(f"inverted BBO spread: {self.instrument}")

    @property
    def mid(self) -> float:
        """Pair-price diagnostic only; never used for account-currency conversion."""
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class ConversionHop:
    instrument: str
    from_currency: str
    to_currency: str

    def __post_init__(self) -> None:
        base, quote = pair_currencies(self.instrument)
        if {self.from_currency, self.to_currency} != {base, quote}:
            raise AccountingError(f"conversion hop does not match pair: {self}")
        if self.from_currency == self.to_currency:
            raise AccountingError("conversion hop cannot be a self-loop")


USD_TO_JPY = ConversionHop("USD_JPY", "USD", "JPY")
CAD_TO_USD = ConversionHop("USD_CAD", "CAD", "USD")
CHF_TO_USD = ConversionHop("USD_CHF", "CHF", "USD")

# Each source currency maps to path *candidates*.  Exactly one candidate must
# exist, making path ambiguity a mechanical stop rather than a price-based
# route choice.
DEFAULT_PATH_CANDIDATES: dict[str, tuple[tuple[ConversionHop, ...], ...]] = {
    "USD": ((USD_TO_JPY,),),
    "CAD": ((CAD_TO_USD, USD_TO_JPY),),
    "CHF": ((CHF_TO_USD, USD_TO_JPY),),
}


@dataclass(frozen=True)
class ConversionResult:
    currency: str
    source_amount: float
    event_time: str
    side: str
    jpy_amount: float
    executable_rate_jpy_per_currency: float
    bid_ask_width_jpy: float
    path: tuple[str, ...]
    quote_evidence: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversionBook:
    """Append-order BBO book with explicit, unique conversion paths."""

    def __init__(
        self,
        events: Iterable[BBO],
        *,
        max_staleness_seconds: int = MAX_CONVERSION_STALENESS_SECONDS,
        path_candidates: Mapping[str, Sequence[Sequence[ConversionHop]]] | None = None,
    ) -> None:
        if not isinstance(max_staleness_seconds, int) or max_staleness_seconds < 0:
            raise AccountingError("conversion staleness must be a nonnegative integer second value")
        self.max_staleness_ns = max_staleness_seconds * 1_000_000_000
        self.path_candidates = {
            currency: tuple(tuple(path) for path in candidates)
            for currency, candidates in (
                DEFAULT_PATH_CANDIDATES if path_candidates is None else path_candidates
            ).items()
        }
        self._events: dict[str, list[BBO]] = {}
        self._times: dict[str, list[int]] = {}
        for event in events:
            values = self._events.setdefault(event.instrument, [])
            times = self._times.setdefault(event.instrument, [])
            stamp = parse_utc_nanoseconds(event.source_time).value
            if times and stamp <= times[-1]:
                raise AccountingError(
                    f"duplicate or reversed BBO chronology: {event.instrument}/{event.source_time}"
                )
            values.append(event)
            times.append(stamp)
        self._validate_paths()

    def _validate_paths(self) -> None:
        for source, candidates in self.path_candidates.items():
            if source == "JPY":
                raise AccountingError("JPY must use the direct no-path conversion")
            if len(candidates) != 1:
                raise AccountingError(f"ambiguous conversion path for {source}")
            path = candidates[0]
            if not path or path[0].from_currency != source or path[-1].to_currency != "JPY":
                raise AccountingError(f"conversion path does not terminate {source}->JPY")
            visited = {source}
            current = source
            for hop in path:
                if hop.from_currency != current or hop.to_currency in visited:
                    raise AccountingError(f"disconnected or cyclic conversion path for {source}")
                visited.add(hop.to_currency)
                current = hop.to_currency

    def _path(self, currency: str) -> tuple[ConversionHop, ...]:
        candidates = self.path_candidates.get(currency)
        if candidates is None:
            raise AccountingError(f"missing conversion path for {currency}")
        if len(candidates) != 1:
            raise AccountingError(f"ambiguous conversion path for {currency}")
        return tuple(candidates[0])

    def quote_at(self, instrument: str, event_time: str) -> tuple[BBO, int]:
        try:
            query = parse_utc_nanoseconds(event_time).value
        except ValueError as error:
            raise AccountingError(f"noncanonical accounting event timestamp: {event_time}") from error
        times = self._times.get(instrument, [])
        index = bisect.bisect_right(times, query) - 1
        if index < 0:
            raise AccountingError(f"missing causal BBO at or before event: {instrument}/{event_time}")
        quote = self._events[instrument][index]
        age_ns = query - times[index]
        if age_ns < 0:
            raise AccountingError(f"future conversion quote selected: {instrument}/{event_time}")
        if age_ns > self.max_staleness_ns:
            raise AccountingError(f"stale conversion quote: {instrument}/{event_time}/{age_ns}")
        return quote, age_ns

    @staticmethod
    def _sell(amount: float, source: str, target: str, quote: BBO) -> float:
        base, counter = pair_currencies(quote.instrument)
        if source == base and target == counter:
            return amount * quote.bid
        if source == counter and target == base:
            return amount / quote.ask
        raise AccountingError(f"BBO cannot execute requested conversion: {source}->{target}")

    def _sell_along(
        self, amount: float, path: Sequence[ConversionHop], event_time: str
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        value = amount
        evidence: list[dict[str, Any]] = []
        for hop in path:
            quote, age_ns = self.quote_at(hop.instrument, event_time)
            value = self._sell(value, hop.from_currency, hop.to_currency, quote)
            evidence.append({
                "instrument": quote.instrument,
                "source_time": quote.source_time,
                "event_time": event_time,
                "age_nanoseconds": age_ns,
                "bid": quote.bid,
                "ask": quote.ask,
                "from_currency": hop.from_currency,
                "to_currency": hop.to_currency,
            })
        return value, tuple(evidence)

    def _asset_rate(
        self, currency: str, event_time: str
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        return self._sell_along(1.0, self._path(currency), event_time)

    def _liability_rate(
        self, currency: str, event_time: str
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        path = self._path(currency)
        reverse = tuple(
            ConversionHop(hop.instrument, hop.to_currency, hop.from_currency)
            for hop in reversed(path)
        )
        source_per_jpy, evidence = self._sell_along(1.0, reverse, event_time)
        if source_per_jpy <= 0.0:
            raise AccountingError(f"invalid liability conversion rate for {currency}")
        return 1.0 / source_per_jpy, evidence

    def bounds_to_jpy(self, amount_abs: float, currency: str, event_time: str) -> dict[str, Any]:
        if not math.isfinite(amount_abs) or amount_abs < 0.0:
            raise AccountingError("conversion amount bound must be finite and nonnegative")
        if currency == "JPY":
            return {
                "asset_liquidation_jpy": amount_abs,
                "liability_buyback_jpy": amount_abs,
                "bid_ask_width_jpy": 0.0,
            }
        asset_rate, _ = self._asset_rate(currency, event_time)
        liability_rate, _ = self._liability_rate(currency, event_time)
        asset = amount_abs * asset_rate
        liability = amount_abs * liability_rate
        width = liability - asset
        tolerance = max(1e-12, liability * 1e-12)
        if width < -tolerance:
            raise AccountingError(f"negative conversion spread width for {currency}")
        return {
            "asset_liquidation_jpy": asset,
            "liability_buyback_jpy": liability,
            "bid_ask_width_jpy": max(width, 0.0),
        }

    def convert_to_jpy(self, amount: float, currency: str, event_time: str) -> ConversionResult:
        if not math.isfinite(amount):
            raise AccountingError("conversion amount must be finite")
        try:
            parse_utc_nanoseconds(event_time)
        except ValueError as error:
            raise AccountingError(f"noncanonical accounting event timestamp: {event_time}") from error
        if amount == 0.0:
            return ConversionResult(currency, 0.0, event_time, "ZERO", 0.0, 1.0, 0.0, (), ())
        if currency == "JPY":
            return ConversionResult(
                currency, amount, event_time,
                "ASSET_SELL" if amount > 0.0 else "LIABILITY_BUYBACK",
                amount, 1.0, 0.0, (), (),
            )
        if amount > 0.0:
            rate, evidence = self._asset_rate(currency, event_time)
            side = "ASSET_SELL"
            jpy_amount = amount * rate
        else:
            rate, evidence = self._liability_rate(currency, event_time)
            side = "LIABILITY_BUYBACK"
            jpy_amount = amount * rate
        bounds = self.bounds_to_jpy(abs(amount), currency, event_time)
        return ConversionResult(
            currency=currency,
            source_amount=amount,
            event_time=event_time,
            side=side,
            jpy_amount=jpy_amount,
            executable_rate_jpy_per_currency=rate,
            bid_ask_width_jpy=bounds["bid_ask_width_jpy"],
            path=tuple(item.instrument for item in self._path(currency)),
            quote_evidence=evidence,
        )

    def roundtrip_retention(self, amount_abs: float, currency: str, event_time: str) -> float:
        if amount_abs <= 0.0 or not math.isfinite(amount_abs):
            raise AccountingError("round-trip amount must be positive and finite")
        if currency == "JPY":
            return 1.0
        sold = self.convert_to_jpy(amount_abs, currency, event_time).jpy_amount
        liability_rate, _ = self._liability_rate(currency, event_time)
        repurchased = sold / liability_rate
        return repurchased / amount_abs


@dataclass(frozen=True)
class CostScenario:
    name: str
    slippage_pips_per_side: float
    commission_bps_per_side: float
    financing_bps_per_day: float
    raw_pair_mid: bool = False

    def __post_init__(self) -> None:
        values = (
            self.slippage_pips_per_side,
            self.commission_bps_per_side,
            self.financing_bps_per_day,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise AccountingError(f"invalid cost scenario: {self.name}")


RAW_SIGNAL = CostScenario("RAW_SIGNAL", 0.0, 0.0, 0.0, True)
EXECUTABLE_BASE = CostScenario("EXECUTABLE_BASE", 0.3, 0.0, 0.5, False)
ADVERSE_STRESS = CostScenario("ADVERSE_STRESS", 0.9, 0.2, 1.5, False)
SCENARIOS = {
    item.name: item for item in (RAW_SIGNAL, EXECUTABLE_BASE, ADVERSE_STRESS)
}


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
    def base_currency(self) -> str:
        return pair_currencies(self.pair)[0]

    @property
    def quote_currency(self) -> str:
        return pair_currencies(self.pair)[1]

    def currency_inventory(self) -> dict[str, float]:
        entry_price = abs(self.sizing_quote_cashflow_per_unit)
        return {
            self.base_currency: self.direction * self.units,
            self.quote_currency: -self.direction * self.units * entry_price,
        }


def _pair_price(
    bbo: BBO, direction: int, *, opening: bool, scenario: CostScenario
) -> float:
    if direction not in (-1, 1):
        raise AccountingError("trade direction must be -1 or +1")
    if scenario.raw_pair_mid:
        return bbo.mid
    slip = scenario.slippage_pips_per_side * pip_size(bbo.instrument)
    if opening:
        price = bbo.ask + slip if direction > 0 else bbo.bid - slip
    else:
        price = bbo.bid - slip if direction > 0 else bbo.ask + slip
    if not math.isfinite(price) or price <= 0.0:
        raise AccountingError(f"slippage produced invalid pair price: {bbo.instrument}")
    return price


def _validate_pair_bbo_chronology(bbo: BBO, event_time: str, role: str) -> None:
    source_ns = parse_utc_nanoseconds(bbo.source_time).value
    event_ns = parse_utc_nanoseconds(event_time).value
    age_ns = event_ns - source_ns
    if age_ns < 0:
        raise AccountingError(f"future {role} BBO")
    if age_ns > MAX_CONVERSION_STALENESS_NS:
        raise AccountingError(f"stale {role} BBO")


def size_position(
    position_id: str,
    pair: str,
    direction: int,
    notional_jpy: float,
    entry_time: str,
    entry_bbo: BBO,
    conversion_book: ConversionBook,
) -> Position:
    if not position_id:
        raise AccountingError("position id is required")
    if entry_bbo.instrument != pair:
        raise AccountingError("entry BBO identity mismatch")
    _validate_pair_bbo_chronology(entry_bbo, entry_time, "entry pair")
    if direction not in (-1, 1):
        raise AccountingError("trade direction must be -1 or +1")
    if not math.isfinite(notional_jpy) or notional_jpy <= 0.0:
        raise AccountingError("JPY notional must be positive and finite")
    entry_price = _pair_price(entry_bbo, direction, opening=True, scenario=CostScenario(
        "SIZING_EXECUTABLE_BBO", 0.0, 0.0, 0.0, False
    ))
    quote_currency = pair_currencies(pair)[1]
    quote_cashflow_per_unit = -direction * entry_price
    converted = conversion_book.convert_to_jpy(
        quote_cashflow_per_unit, quote_currency, entry_time
    )
    value_per_unit = abs(converted.jpy_amount)
    if value_per_unit <= 0.0:
        raise AccountingError("entry conversion produced zero JPY unit value")
    units = notional_jpy / value_per_unit
    if not math.isfinite(units) or units <= 0.0:
        raise AccountingError("position sizing produced invalid units")
    return Position(
        position_id=position_id,
        pair=pair,
        direction=direction,
        units=units,
        notional_jpy=notional_jpy,
        entry_time=entry_time,
        entry_bbo=entry_bbo,
        sizing_quote_cashflow_per_unit=quote_cashflow_per_unit,
        sizing_jpy_cashflow_per_unit=converted.jpy_amount,
    )


def evaluate_position(
    position: Position,
    exit_time: str,
    exit_bbo: BBO,
    conversion_book: ConversionBook,
    scenario: CostScenario,
) -> dict[str, Any]:
    if exit_bbo.instrument != position.pair:
        raise AccountingError("exit BBO identity mismatch")
    _validate_pair_bbo_chronology(exit_bbo, exit_time, "exit pair")
    entry_ns = parse_utc_nanoseconds(position.entry_time).value
    exit_ns = parse_utc_nanoseconds(exit_time).value
    if exit_ns < entry_ns:
        raise AccountingError("position cannot be evaluated before entry")
    elapsed_seconds = (exit_ns - entry_ns) / 1_000_000_000
    elapsed_days = elapsed_seconds / 86_400.0
    direction, units = position.direction, position.units
    quote_currency = position.quote_currency

    raw_entry = position.entry_bbo.mid
    raw_exit = exit_bbo.mid
    raw_quote_pnl = direction * units * (raw_exit - raw_entry)
    gross = conversion_book.convert_to_jpy(raw_quote_pnl, quote_currency, exit_time)

    entry_price = _pair_price(
        position.entry_bbo, direction, opening=True, scenario=scenario
    )
    exit_price = _pair_price(exit_bbo, direction, opening=False, scenario=scenario)
    pair_quote_pnl = direction * units * (exit_price - entry_price)
    pair_converted = conversion_book.convert_to_jpy(pair_quote_pnl, quote_currency, exit_time)
    pair_at_entry_conversion = conversion_book.convert_to_jpy(
        pair_quote_pnl, quote_currency, position.entry_time
    )

    entry_commission_quote = units * entry_price * scenario.commission_bps_per_side * 1e-4
    exit_commission_quote = units * exit_price * scenario.commission_bps_per_side * 1e-4
    financing_quote = (
        units * entry_price * scenario.financing_bps_per_day * 1e-4 * elapsed_days
    )
    entry_commission = conversion_book.convert_to_jpy(
        -entry_commission_quote, quote_currency, position.entry_time
    )
    exit_commission = conversion_book.convert_to_jpy(
        -exit_commission_quote, quote_currency, exit_time
    )
    financing = conversion_book.convert_to_jpy(
        -financing_quote, quote_currency, exit_time
    )
    commission_cost_jpy = -(entry_commission.jpy_amount + exit_commission.jpy_amount)
    financing_cost_jpy = -financing.jpy_amount
    net_jpy = pair_converted.jpy_amount - commission_cost_jpy - financing_cost_jpy
    total_cost_jpy = gross.jpy_amount - net_jpy
    bounds = conversion_book.bounds_to_jpy(abs(pair_quote_pnl), quote_currency, exit_time)

    payload = {
        "position_id": position.position_id,
        "pair": position.pair,
        "direction": direction,
        "units": units,
        "notional_jpy": position.notional_jpy,
        "entry_time": position.entry_time,
        "exit_or_mark_time": exit_time,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_financing_days": elapsed_days,
        "scenario": scenario.name,
        "raw_pair_entry_mid": raw_entry,
        "raw_pair_exit_mid": raw_exit,
        "executable_entry_price": entry_price,
        "executable_exit_price": exit_price,
        "raw_quote_pnl": raw_quote_pnl,
        "executable_pair_quote_pnl": pair_quote_pnl,
        "gross_jpy": gross.jpy_amount,
        "executable_pair_jpy": pair_converted.jpy_amount,
        "commission_cost_jpy": commission_cost_jpy,
        "financing_cost_jpy": financing_cost_jpy,
        "net_jpy": net_jpy,
        "gross_return_on_fixed_notional": gross.jpy_amount / position.notional_jpy,
        "net_return_on_fixed_notional": net_jpy / position.notional_jpy,
        "pair_spread_slippage_cost_jpy": gross.jpy_amount - pair_converted.jpy_amount,
        "total_realized_cost_jpy": total_cost_jpy,
        "conversion_move_jpy": (
            pair_converted.jpy_amount - pair_at_entry_conversion.jpy_amount
        ),
        "conversion_bid_ask_width_jpy": bounds["bid_ask_width_jpy"],
        "gross_conversion": gross.as_dict(),
        "pair_pnl_conversion": pair_converted.as_dict(),
        "entry_commission_conversion": entry_commission.as_dict(),
        "exit_commission_conversion": exit_commission.as_dict(),
        "financing_conversion": financing.as_dict(),
        "currency_inventory_before_close": position.currency_inventory(),
        "currency_inventory_after_close": {},
        "terminal_inventory_mtm_jpy": 0.0,
        "linear_long_short_pnl": True,
        "account_currency_midpoint_conversion_used": False,
    }
    payload["evaluation_sha256"] = embedded_hash(payload, "evaluation_sha256")
    return payload


class JPYAccountLedger:
    """Append-only paper account state for one fixed cost arm."""

    def __init__(
        self,
        initial_cash_jpy: float,
        conversion_book: ConversionBook,
        scenario: CostScenario,
    ) -> None:
        if not math.isfinite(initial_cash_jpy) or initial_cash_jpy <= 0.0:
            raise AccountingError("initial JPY cash must be positive and finite")
        self.initial_cash_jpy = initial_cash_jpy
        self.cash_jpy = initial_cash_jpy
        self.conversion_book = conversion_book
        self.scenario = scenario
        self.positions: dict[str, Position] = {}
        self.events: list[dict[str, Any]] = []
        self._last_event_ns: int | None = None

    def _append(self, event_time: str, event: dict[str, Any]) -> None:
        stamp = parse_utc_nanoseconds(event_time).value
        if self._last_event_ns is not None and stamp < self._last_event_ns:
            raise AccountingError("account ledger event chronology reversed")
        row = {"sequence": len(self.events) + 1, "event_time": event_time, **event}
        row["event_sha256"] = embedded_hash(row, "event_sha256")
        self.events.append(row)
        self._last_event_ns = stamp

    def open(
        self,
        position_id: str,
        pair: str,
        direction: int,
        notional_jpy: float,
        event_time: str,
        pair_bbo: BBO,
    ) -> Position:
        if position_id in self.positions:
            raise AccountingError(f"position already exists: {position_id}")
        position = size_position(
            position_id, pair, direction, notional_jpy, event_time,
            pair_bbo, self.conversion_book,
        )
        self.positions[position_id] = position
        self._append(event_time, {
            "event_type": "OPEN",
            "position_id": position_id,
            "pair": pair,
            "direction": direction,
            "units": position.units,
            "notional_jpy": notional_jpy,
            "currency_inventory_after": position.currency_inventory(),
            "cash_jpy_after": self.cash_jpy,
        })
        return position

    def mark(self, position_id: str, event_time: str, pair_bbo: BBO) -> dict[str, Any]:
        position = self.positions.get(position_id)
        if position is None:
            raise AccountingError(f"missing position for mark: {position_id}")
        evaluation = evaluate_position(
            position, event_time, pair_bbo, self.conversion_book, self.scenario
        )
        self._append(event_time, {
            "event_type": "MARK",
            "position_id": position_id,
            "evaluation_sha256": evaluation["evaluation_sha256"],
            "conservative_liquidation_mtm_jpy": evaluation["net_jpy"],
            "cash_jpy_after": self.cash_jpy,
        })
        return evaluation

    def close(
        self,
        position_id: str,
        event_time: str,
        pair_bbo: BBO,
        *,
        reason: str,
    ) -> dict[str, Any]:
        position = self.positions.get(position_id)
        if position is None:
            raise AccountingError(f"missing position for close: {position_id}")
        evaluation = evaluate_position(
            position, event_time, pair_bbo, self.conversion_book, self.scenario
        )
        self.cash_jpy += evaluation["net_jpy"]
        del self.positions[position_id]
        self._append(event_time, {
            "event_type": "CLOSE",
            "position_id": position_id,
            "reason": reason,
            "evaluation_sha256": evaluation["evaluation_sha256"],
            "realized_net_jpy": evaluation["net_jpy"],
            "currency_inventory_after": {},
            "cash_jpy_after": self.cash_jpy,
        })
        return evaluation

    def portfolio_mark(self, event_time: str, pair_bbos: Mapping[str, BBO]) -> dict[str, Any]:
        evaluations = []
        for position_id in sorted(self.positions):
            position = self.positions[position_id]
            if position.pair not in pair_bbos:
                raise AccountingError(f"missing pair mark BBO: {position.pair}")
            evaluations.append(evaluate_position(
                position, event_time, pair_bbos[position.pair],
                self.conversion_book, self.scenario,
            ))
        total_mtm = sum(item["net_jpy"] for item in evaluations)
        return {
            "event_time": event_time,
            "cash_jpy": self.cash_jpy,
            "open_position_mtm_jpy": total_mtm,
            "equity_jpy": self.cash_jpy + total_mtm,
            "position_count": len(evaluations),
            "position_evaluation_sha256": hashlib.sha256(canonical_bytes(
                [item["evaluation_sha256"] for item in evaluations]
            )).hexdigest(),
        }

    def terminal_liquidate(
        self,
        event_time: str,
        pair_bbos: Mapping[str, BBO],
    ) -> dict[str, Any]:
        evaluations = []
        for position_id in sorted(tuple(self.positions)):
            pair = self.positions[position_id].pair
            if pair not in pair_bbos:
                raise AccountingError(f"missing terminal BBO: {pair}")
            evaluations.append(self.close(
                position_id, event_time, pair_bbos[pair], reason="TERMINAL_LIQUIDATION"
            ))
        if self.positions:
            raise AccountingError("terminal liquidation left open inventory")
        return {
            "event_time": event_time,
            "terminal_open_inventory": 0,
            "terminal_inventory_mtm_jpy": 0.0,
            "ending_cash_jpy": self.cash_jpy,
            "realized_net_jpy": sum(item["net_jpy"] for item in evaluations),
            "liquidated_positions": len(evaluations),
            "ledger_sha256": self.ledger_sha256(),
        }

    def realized_by_exit_month(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for event in self.events:
            if event["event_type"] != "CLOSE":
                continue
            month = event["event_time"][:7]
            result[month] = result.get(month, 0.0) + event["realized_net_jpy"]
        return dict(sorted(result.items()))

    def ledger_sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self.events)).hexdigest()
