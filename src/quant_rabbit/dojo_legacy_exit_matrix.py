"""Causal exit-policy comparison for recovered legacy DOJO entry cohorts.

This module is deliberately broker-free.  It consumes timestamped executable
bid/ask quotes and already-recovered entry signals, then compares exit policies
against the exact same accepted entry cohort and cost assumptions.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import datetime
from math import inf
from typing import Iterable, Literal, Sequence


Side = Literal["long", "short"]
PolicyName = Literal[
    "no_sl",
    "fixed_sl",
    "atr_sl",
    "volatility_trail",
    "time_stop",
]


@dataclass(frozen=True)
class Quote:
    timestamp: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class EntrySignal:
    signal_id: str
    timestamp: datetime
    side: Side
    atr_pips: float
    take_profit_pips: float


@dataclass(frozen=True)
class ReplayCosts:
    units: int = 1_000
    slippage_pips_per_fill: float = 0.05
    financing_jpy_per_10k_units_per_day: float = 0.0
    ai_cost_jpy_per_decision: float = 0.0


@dataclass(frozen=True)
class ExitPolicy:
    name: PolicyName
    fixed_stop_pips: float = 5.0
    atr_stop_multiple: float = 1.5
    trail_atr_multiple: float = 1.0
    time_stop_seconds: int = 300


@dataclass(frozen=True)
class InventoryPolicy:
    enabled: bool
    checkpoint_seconds: int = 60


@dataclass(frozen=True)
class ClosedTrade:
    signal_id: str
    side: Side
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    exit_reason: str
    gross_jpy: float
    financing_jpy: float
    ai_cost_jpy: float
    net_jpy: float


@dataclass(frozen=True)
class ReplayMetrics:
    policy: PolicyName
    ai_inventory: bool
    entry_cohort_size: int
    trades: int
    net_jpy: float
    profit_factor: float | None
    expectancy_jpy: float | None
    max_drawdown_jpy: float
    ai_decisions: int
    ai_cost_jpy: float
    profitable: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayArmResult:
    metrics: ReplayMetrics
    trades: tuple[ClosedTrade, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "trades": [asdict(trade) for trade in self.trades],
        }


def _price_move_pips(side: Side, entry: float, exit_price: float) -> float:
    direction = 1.0 if side == "long" else -1.0
    return direction * (exit_price - entry) / 0.01


def _pips_to_jpy(pips: float, units: int) -> float:
    return round(pips * 0.01 * units, 8)


def _entry_price(quote: Quote, side: Side, slippage_pips: float) -> float:
    slippage = slippage_pips * 0.01
    return quote.ask + slippage if side == "long" else quote.bid - slippage


def _exit_price(quote: Quote, side: Side, slippage_pips: float) -> float:
    slippage = slippage_pips * 0.01
    return quote.bid - slippage if side == "long" else quote.ask + slippage


def _should_inventory_exit(
    *,
    signal: EntrySignal,
    entry_price: float,
    quote: Quote,
    previous_quote: Quote | None,
    checkpoint_seconds: int,
) -> bool:
    """Frozen causal inventory rule using only information available at quote."""

    age_seconds = (quote.timestamp - signal.timestamp).total_seconds()
    if age_seconds < checkpoint_seconds or previous_quote is None:
        return False
    pnl_pips = _price_move_pips(
        signal.side,
        entry_price,
        quote.bid if signal.side == "long" else quote.ask,
    )
    recent_move = quote.mid - previous_quote.mid
    adverse_momentum = (
        recent_move < 0.0 if signal.side == "long" else recent_move > 0.0
    )
    return pnl_pips < 0.0 and adverse_momentum


def replay_arm(
    *,
    quotes: Sequence[Quote],
    entries: Sequence[EntrySignal],
    policy: ExitPolicy,
    costs: ReplayCosts,
    inventory: InventoryPolicy,
) -> ReplayArmResult:
    """Replay one arm with every entry evaluated as an independent position.

    Independent position ownership preserves the exact same entry cohort across
    arms even when their exit timestamps differ.
    """

    if not quotes:
        raise ValueError("quotes must not be empty")
    if any(quotes[index].timestamp > quotes[index + 1].timestamp for index in range(len(quotes) - 1)):
        raise ValueError("quotes must be timestamp-sorted")

    ordered_entries = tuple(sorted(entries, key=lambda item: (item.timestamp, item.signal_id)))
    quote_timestamps = tuple(quote.timestamp for quote in quotes)
    closed: list[ClosedTrade] = []
    ai_decisions = 0
    cumulative_ai_cost = 0.0

    for signal in ordered_entries:
        entry_index = bisect_left(quote_timestamps, signal.timestamp)
        if entry_index >= len(quotes):
            break
        entry_quote = quotes[entry_index]
        entry = _entry_price(entry_quote, signal.side, costs.slippage_pips_per_fill)
        stop_pips: float | None
        if policy.name == "fixed_sl":
            stop_pips = policy.fixed_stop_pips
        elif policy.name in {"atr_sl", "volatility_trail"}:
            stop_pips = max(signal.atr_pips * policy.atr_stop_multiple, 0.1)
        else:
            stop_pips = None

        best_favorable_pips = -inf
        exit_quote = quotes[-1]
        exit_reason = "period_end_mtm"
        previous_quote: Quote | None = None

        for quote_index in range(entry_index, len(quotes)):
            quote = quotes[quote_index]
            executable_exit = quote.bid if signal.side == "long" else quote.ask
            move_pips = _price_move_pips(signal.side, entry, executable_exit)
            best_favorable_pips = max(best_favorable_pips, move_pips)
            age_seconds = (quote.timestamp - entry_quote.timestamp).total_seconds()

            if move_pips >= signal.take_profit_pips:
                exit_quote = quote
                exit_reason = "take_profit"
                break
            if stop_pips is not None and move_pips <= -stop_pips:
                exit_quote = quote
                exit_reason = "stop_loss"
                break
            if policy.name == "volatility_trail":
                trail_distance = max(
                    signal.atr_pips * policy.trail_atr_multiple, 0.1
                )
                if (
                    best_favorable_pips > 0.0
                    and move_pips <= best_favorable_pips - trail_distance
                ):
                    exit_quote = quote
                    exit_reason = "volatility_trail"
                    break
            if (
                policy.name == "time_stop"
                and age_seconds >= policy.time_stop_seconds
            ):
                exit_quote = quote
                exit_reason = "time_stop"
                break
            if inventory.enabled and age_seconds >= inventory.checkpoint_seconds:
                ai_decisions += 1
                if _should_inventory_exit(
                    signal=signal,
                    entry_price=entry,
                    quote=quote,
                    previous_quote=previous_quote,
                    checkpoint_seconds=inventory.checkpoint_seconds,
                ):
                    exit_quote = quote
                    exit_reason = "inventory_exit"
                    break
            previous_quote = quote

        exit_price = _exit_price(
            exit_quote, signal.side, costs.slippage_pips_per_fill
        )
        gross_jpy = _pips_to_jpy(
            _price_move_pips(signal.side, entry, exit_price), costs.units
        )
        holding_days = max(
            (exit_quote.timestamp - entry_quote.timestamp).total_seconds(), 0.0
        ) / 86_400.0
        financing_jpy = round(
            costs.financing_jpy_per_10k_units_per_day
            * (costs.units / 10_000.0)
            * holding_days,
            8,
        )
        decision_cost = (
            costs.ai_cost_jpy_per_decision * ai_decisions
            if inventory.enabled
            else 0.0
        )
        ai_cost_jpy = round(max(decision_cost - cumulative_ai_cost, 0.0), 8)
        cumulative_ai_cost = round(cumulative_ai_cost + ai_cost_jpy, 8)
        net_jpy = round(gross_jpy - financing_jpy - ai_cost_jpy, 8)
        closed.append(
            ClosedTrade(
                signal_id=signal.signal_id,
                side=signal.side,
                entry_timestamp=entry_quote.timestamp,
                exit_timestamp=exit_quote.timestamp,
                entry_price=entry,
                exit_price=exit_price,
                exit_reason=exit_reason,
                gross_jpy=gross_jpy,
                financing_jpy=financing_jpy,
                ai_cost_jpy=ai_cost_jpy,
                net_jpy=net_jpy,
            )
        )
    wins = sum(trade.net_jpy for trade in closed if trade.net_jpy > 0.0)
    losses = -sum(trade.net_jpy for trade in closed if trade.net_jpy < 0.0)
    profit_factor = wins / losses if losses > 0.0 else (None if wins == 0.0 else inf)
    net_jpy = round(sum(trade.net_jpy for trade in closed), 8)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in closed:
        equity += trade.net_jpy
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    metrics = ReplayMetrics(
        policy=policy.name,
        ai_inventory=inventory.enabled,
        entry_cohort_size=len(ordered_entries),
        trades=len(closed),
        net_jpy=net_jpy,
        profit_factor=profit_factor,
        expectancy_jpy=(net_jpy / len(closed)) if closed else None,
        max_drawdown_jpy=round(max_drawdown, 8),
        ai_decisions=ai_decisions,
        ai_cost_jpy=round(sum(trade.ai_cost_jpy for trade in closed), 8),
        profitable=net_jpy > 0.0,
    )
    return ReplayArmResult(metrics=metrics, trades=tuple(closed))


def replay_exit_matrix(
    *,
    quotes: Iterable[Quote],
    entries: Iterable[EntrySignal],
    costs: ReplayCosts,
    policies: Sequence[ExitPolicy] | None = None,
    inventory_policies: Sequence[InventoryPolicy] | None = None,
) -> tuple[ReplayArmResult, ...]:
    quote_sequence = tuple(quotes)
    entry_sequence = tuple(entries)
    selected_policies = policies or (
        ExitPolicy("no_sl"),
        ExitPolicy("fixed_sl"),
        ExitPolicy("atr_sl"),
        ExitPolicy("volatility_trail"),
        ExitPolicy("time_stop"),
    )
    selected_inventory = inventory_policies or (
        InventoryPolicy(False),
        InventoryPolicy(True),
    )
    return tuple(
        replay_arm(
            quotes=quote_sequence,
            entries=entry_sequence,
            policy=policy,
            costs=costs,
            inventory=inventory,
        )
        for policy in selected_policies
        for inventory in selected_inventory
    )
