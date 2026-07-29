from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

from .config import CryptoSafetyContract
from .ledger import CryptoLedger


def _d(value: object) -> Decimal:
    return Decimal(str(value))


def _s(value: Decimal) -> str:
    return format(value, "f")


@dataclass
class PaperState:
    initial_cash_jpy: Decimal
    cash_jpy: Decimal
    positions: dict[str, Decimal] = field(default_factory=dict)
    average_costs: dict[str, Decimal] = field(default_factory=dict)
    position_regimes: dict[str, str] = field(default_factory=dict)
    open_trade_contexts: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    realized_pnl_by_pair: dict[str, Decimal] = field(default_factory=dict)
    realized_pnl_by_regime: dict[str, Decimal] = field(default_factory=dict)
    gross_profit_jpy: Decimal = Decimal("0")
    gross_loss_jpy: Decimal = Decimal("0")
    round_trips: int = 0
    interest_cost_jpy: Decimal = Decimal("0")
    interest_cost_by_pair: dict[str, Decimal] = field(default_factory=dict)
    margin_calls: int = 0
    forced_liquidations: int = 0
    fees_jpy: Decimal = Decimal("0")
    spread_cost_jpy: Decimal = Decimal("0")
    slippage_cost_jpy: Decimal = Decimal("0")
    fills: int = 0
    partial_fills: int = 0
    unfilled_orders: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    turnover_jpy: Decimal = Decimal("0")
    discipline_violations: int = 0
    peak_equity_jpy: Decimal = Decimal("0")
    max_drawdown_jpy: Decimal = Decimal("0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_cash_jpy": _s(self.initial_cash_jpy),
            "cash_jpy": _s(self.cash_jpy),
            "positions": {key: _s(value) for key, value in self.positions.items()},
            "average_costs": {
                key: _s(value) for key, value in self.average_costs.items()
            },
            "position_regimes": dict(self.position_regimes),
            "open_trade_contexts": dict(self.open_trade_contexts),
            "realized_pnl_by_pair": {
                key: _s(value)
                for key, value in self.realized_pnl_by_pair.items()
            },
            "realized_pnl_by_regime": {
                key: _s(value)
                for key, value in self.realized_pnl_by_regime.items()
            },
            "gross_profit_jpy": _s(self.gross_profit_jpy),
            "gross_loss_jpy": _s(self.gross_loss_jpy),
            "round_trips": self.round_trips,
            "interest_cost_jpy": _s(self.interest_cost_jpy),
            "interest_cost_by_pair": {
                key: _s(value)
                for key, value in self.interest_cost_by_pair.items()
            },
            "margin_calls": self.margin_calls,
            "forced_liquidations": self.forced_liquidations,
            "fees_jpy": _s(self.fees_jpy),
            "spread_cost_jpy": _s(self.spread_cost_jpy),
            "slippage_cost_jpy": _s(self.slippage_cost_jpy),
            "fills": self.fills,
            "partial_fills": self.partial_fills,
            "unfilled_orders": self.unfilled_orders,
            "maker_fills": self.maker_fills,
            "taker_fills": self.taker_fills,
            "turnover_jpy": _s(self.turnover_jpy),
            "discipline_violations": self.discipline_violations,
            "peak_equity_jpy": _s(self.peak_equity_jpy),
            "max_drawdown_jpy": _s(self.max_drawdown_jpy),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PaperState":
        return cls(
            initial_cash_jpy=_d(raw["initial_cash_jpy"]),
            cash_jpy=_d(raw["cash_jpy"]),
            positions={key: _d(value) for key, value in raw["positions"].items()},
            average_costs={
                key: _d(value) for key, value in raw["average_costs"].items()
            },
            position_regimes={
                key: str(value)
                for key, value in raw.get("position_regimes", {}).items()
            },
            open_trade_contexts={
                key: dict(value)
                for key, value in raw.get("open_trade_contexts", {}).items()
            },
            realized_pnl_by_pair={
                key: _d(value)
                for key, value in raw.get("realized_pnl_by_pair", {}).items()
            },
            realized_pnl_by_regime={
                key: _d(value)
                for key, value in raw.get("realized_pnl_by_regime", {}).items()
            },
            gross_profit_jpy=_d(raw.get("gross_profit_jpy", "0")),
            gross_loss_jpy=_d(raw.get("gross_loss_jpy", "0")),
            round_trips=int(raw.get("round_trips", 0)),
            interest_cost_jpy=_d(raw.get("interest_cost_jpy", "0")),
            interest_cost_by_pair={
                key: _d(value)
                for key, value in raw.get(
                    "interest_cost_by_pair", {}
                ).items()
            },
            margin_calls=int(raw.get("margin_calls", 0)),
            forced_liquidations=int(raw.get("forced_liquidations", 0)),
            fees_jpy=_d(raw["fees_jpy"]),
            spread_cost_jpy=_d(raw["spread_cost_jpy"]),
            slippage_cost_jpy=_d(raw["slippage_cost_jpy"]),
            fills=int(raw["fills"]),
            partial_fills=int(raw["partial_fills"]),
            unfilled_orders=int(raw["unfilled_orders"]),
            maker_fills=int(raw.get("maker_fills", 0)),
            taker_fills=int(raw.get("taker_fills", 0)),
            turnover_jpy=_d(raw.get("turnover_jpy", "0")),
            discipline_violations=int(raw["discipline_violations"]),
            peak_equity_jpy=_d(raw["peak_equity_jpy"]),
            max_drawdown_jpy=_d(raw["max_drawdown_jpy"]),
        )


class PaperEngine:
    """Conservative spot/margin Paper engine with no broker dependency.

    Spot mode makes shorting unreachable. Margin mode permits modeled long and
    short positions up to 2x while retaining authority NONE.
    """

    def __init__(
        self,
        ledger: CryptoLedger,
        *,
        initial_cash_jpy: Decimal = Decimal("10000"),
        maker_fill_fraction: Decimal = Decimal("0.25"),
        allow_short: bool = False,
        max_leverage: Decimal = Decimal("1"),
        margin_call_ratio: Decimal = Decimal("0.50"),
        maintenance_margin_ratio: Decimal = Decimal("0.25"),
        trade_sink: Callable[[dict[str, Any]], object] | None = None,
    ) -> None:
        CryptoSafetyContract.from_env().assert_safe()
        self.ledger = ledger
        restored = ledger.latest_payload("PAPER_STATE")
        self.state = (
            PaperState.from_dict(restored)
            if restored
            else PaperState(
                initial_cash_jpy=initial_cash_jpy,
                cash_jpy=initial_cash_jpy,
                peak_equity_jpy=initial_cash_jpy,
            )
        )
        if not restored or "turnover_jpy" not in restored:
            self.state.turnover_jpy = sum(
                (
                    _d(row["payload"].get("filled_amount", "0"))
                    * _d(row["payload"].get("average_price", "0"))
                    for row in ledger.events("PAPER_FILL")
                    if row["payload"].get("status") != "UNFILLED"
                ),
                Decimal("0"),
            )
        self.maker_fill_fraction = maker_fill_fraction
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.margin_call_ratio = margin_call_ratio
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.trade_sink = trade_sink
        if self.max_leverage < 1 or self.max_leverage > 2:
            raise ValueError("Paper leverage must stay within 1x..2x")
        if not (
            0
            < self.maintenance_margin_ratio
            < self.margin_call_ratio
            < 1
        ):
            raise ValueError("Paper margin thresholds must satisfy 0 < loss < call < 1")

    def process_intent(
        self,
        intent: dict[str, Any],
        *,
        depth: dict[str, Any],
        maker_fee_rate: Decimal,
        taker_fee_rate: Decimal,
    ) -> dict[str, Any]:
        if intent.get("authority") != "NONE" or intent.get("live_permission"):
            raise RuntimeError("paper engine rejected intent with non-NONE authority")
        side = str(intent.get("side", "")).upper()
        if side not in {"BUY", "SELL"}:
            raise RuntimeError("paper engine permits BUY or SELL")
        pair = str(intent["pair"])
        previous_amount = self.state.positions.get(pair, Decimal("0"))
        position_effect = str(
            intent.get(
                "position_effect",
                "OPEN" if side == "BUY" else "CLOSE",
            )
        ).upper()
        if position_effect not in {"OPEN", "CLOSE"}:
            raise RuntimeError("position_effect must be OPEN or CLOSE")
        if position_effect == "OPEN":
            position_side = "LONG" if side == "BUY" else "SHORT"
            if position_side == "SHORT" and not self.allow_short:
                raise RuntimeError("short opening is disabled for spot Paper")
            if (position_side == "LONG" and previous_amount < 0) or (
                position_side == "SHORT" and previous_amount > 0
            ):
                raise RuntimeError("Paper engine forbids hedging or position flips")
        else:
            position_side = "LONG" if side == "SELL" else "SHORT"
        intent_id = str(intent["intent_id"])
        previous_fill = self.ledger.payload_for_dedupe(f"paper-fill:{intent_id}")
        if previous_fill is not None:
            return previous_fill
        requested_amount = _d(intent["amount"])
        order_style = str(intent["order_style"])
        regime = str(intent.get("regime", "UNKNOWN"))
        order_payload = {
            "intent_id": intent_id,
            "pair": pair,
            "side": side,
            "position_effect": position_effect,
            "position_side": position_side,
            "requested_amount": _s(requested_amount),
            "order_style": order_style,
            "regime": regime,
            "paper_only": True,
        }
        self.ledger.append(
            "PAPER_ORDER",
            intent_id,
            order_payload,
            dedupe_key=f"paper-order:{intent_id}",
        )
        if order_style == "PAPER_TAKER":
            levels = (
                depth.get("asks", [])
                if side == "BUY"
                else depth.get("bids", [])
            )
            fill_amount, fill_price, best_price = self._consume_levels(
                levels, requested_amount
            )
            fee_rate = taker_fee_rate
        else:
            levels = (
                depth.get("bids", [])
                if side == "BUY"
                else depth.get("asks", [])
            )
            best_price = _d(levels[0][0]) if levels else Decimal("0")
            available = _d(levels[0][1]) if levels else Decimal("0")
            fill_amount = min(
                requested_amount * self.maker_fill_fraction,
                available * Decimal("0.10"),
            )
            fill_price = best_price
            fee_rate = maker_fee_rate
        if position_effect == "CLOSE":
            closable = (
                previous_amount
                if position_side == "LONG" and previous_amount > 0
                else -previous_amount
                if position_side == "SHORT" and previous_amount < 0
                else Decimal("0")
            )
            fill_amount = max(Decimal("0"), min(fill_amount, closable))
        else:
            capacity = self._opening_notional_capacity()
            fill_amount = max(
                Decimal("0"),
                min(
                    fill_amount,
                    capacity / fill_price
                    if fill_price > 0
                    else Decimal("0"),
                ),
            )
        if (
            side == "BUY"
            and position_effect == "OPEN"
            and not self.allow_short
        ):
            affordable = (
                self.state.cash_jpy / fill_price if fill_price > 0 else Decimal("0")
            )
            fill_amount = max(Decimal("0"), min(fill_amount, affordable))
        if fill_amount <= 0:
            self.state.unfilled_orders += 1
            result = {
                **order_payload,
                "status": "UNFILLED",
                "filled_amount": "0",
                "remaining_amount": _s(requested_amount),
            }
            self.ledger.append(
                "PAPER_FILL",
                intent_id,
                result,
                dedupe_key=f"paper-fill:{intent_id}",
            )
            self._persist_state(intent_id)
            return result
        notional = fill_amount * fill_price
        fee = notional * fee_rate
        cash_delta = notional + fee
        if (
            side == "BUY"
            and position_effect == "OPEN"
            and not self.allow_short
            and cash_delta > self.state.cash_jpy
        ):
            fill_amount = max(
                Decimal("0"),
                self.state.cash_jpy / (fill_price * (Decimal("1") + fee_rate)),
            )
            notional = fill_amount * fill_price
            fee = notional * fee_rate
            cash_delta = notional + fee
        previous_cost = self.state.average_costs.get(pair, Decimal("0"))
        open_context = dict(
            self.state.open_trade_contexts.get(pair, {})
        )
        realized: Decimal | None = None
        if position_effect == "OPEN" and position_side == "LONG":
            new_amount = previous_amount + fill_amount
            average_cost = (
                (previous_amount * previous_cost + notional + fee) / new_amount
                if new_amount > 0
                else Decimal("0")
            )
            self.state.cash_jpy -= cash_delta
            self.state.positions[pair] = new_amount
            self.state.average_costs[pair] = average_cost
            self.state.position_regimes[pair] = regime
        elif position_effect == "OPEN":
            previous_size = -previous_amount
            new_size = previous_size + fill_amount
            proceeds = notional - fee
            average_cost = (
                (previous_size * previous_cost + proceeds) / new_size
                if new_size > 0
                else Decimal("0")
            )
            self.state.cash_jpy += proceeds
            self.state.positions[pair] = -new_size
            self.state.average_costs[pair] = average_cost
            self.state.position_regimes[pair] = regime
        elif position_side == "LONG":
            new_amount = max(Decimal("0"), previous_amount - fill_amount)
            proceeds = notional - fee
            realized = proceeds - (previous_cost * fill_amount)
            self.state.cash_jpy += proceeds
            self.state.positions[pair] = new_amount
            self._record_realized(pair, regime, realized, new_amount)
        else:
            new_amount = min(Decimal("0"), previous_amount + fill_amount)
            close_cost = notional + fee
            realized = (previous_cost * fill_amount) - close_cost
            self.state.cash_jpy -= close_cost
            self.state.positions[pair] = new_amount
            self._record_realized(pair, regime, realized, new_amount)
        self.state.fees_jpy += fee
        spread_cost = (
            max(Decimal("0"), (fill_price - best_price) * fill_amount)
            if side == "BUY"
            else max(Decimal("0"), (best_price - fill_price) * fill_amount)
        )
        slippage_cost = spread_cost
        self.state.spread_cost_jpy += spread_cost
        self.state.slippage_cost_jpy += slippage_cost
        self.state.fills += 1
        self.state.turnover_jpy += notional
        if order_style == "PAPER_TAKER":
            self.state.taker_fills += 1
        else:
            self.state.maker_fills += 1
        remaining = max(Decimal("0"), requested_amount - fill_amount)
        status = "FILLED" if remaining == 0 else "PARTIALLY_FILLED"
        if remaining > 0:
            self.state.partial_fills += 1
        event_at_utc = str(
            intent.get("event_at_utc")
            or datetime.now(timezone.utc).isoformat()
        )
        if position_effect == "OPEN":
            self._record_open_context(
                pair=pair,
                intent=intent,
                position_side=position_side,
                event_at_utc=event_at_utc,
                fill_amount=fill_amount,
                fill_price=fill_price,
                notional=notional,
                fee=fee,
                spread_cost=spread_cost,
                adverse_cost=slippage_cost,
            )
        elif realized is not None:
            open_context = self._record_close_context(
                pair=pair,
                context=open_context,
                event_at_utc=event_at_utc,
                fill_amount=fill_amount,
                fill_price=fill_price,
                realized=realized,
                fee=fee,
                spread_cost=spread_cost,
                adverse_cost=slippage_cost,
                exit_reason=str(
                    intent.get(
                        "signal_reason", intent.get("regime", "UNKNOWN")
                    )
                ),
            )
        result = {
            **order_payload,
            "status": status,
            "filled_amount": _s(fill_amount),
            "remaining_amount": _s(remaining),
            "average_price": _s(fill_price),
            "fee_jpy": _s(fee),
            "fee_rate": _s(fee_rate),
            "spread_cost_jpy": _s(spread_cost),
            "slippage_cost_jpy": _s(slippage_cost),
        }
        self.ledger.append(
            "PAPER_FILL",
            intent_id,
            result,
            dedupe_key=f"paper-fill:{intent_id}",
        )
        self._persist_state(intent_id)
        if (
            position_effect == "CLOSE"
            and new_amount == 0
            and realized is not None
        ):
            closed_trade = self._closed_trade_payload(
                pair=pair,
                intent=intent,
                context=open_context,
                position_side=position_side,
                event_at_utc=event_at_utc,
                fill_amount=fill_amount,
                fill_price=fill_price,
                realized=realized,
                close_fee=fee,
                close_spread=spread_cost,
                close_adverse=slippage_cost,
            )
            trade_id = str(closed_trade["trade_id"])
            dedupe_key = f"paper-trade-closed:{trade_id}"
            self.ledger.append(
                "PAPER_TRADE_CLOSED",
                trade_id,
                closed_trade,
                dedupe_key=dedupe_key,
            )
            metadata = self.ledger.metadata_for_dedupe(dedupe_key) or {}
            outbox_payload = {**closed_trade}
            outbox_payload.update(
                {
                    "ledger_sequence": metadata.get("sequence"),
                    "ledger_event_hash": metadata.get("event_hash"),
                    "ledger_prev_hash": metadata.get("prev_hash"),
                }
            )
            self.state.open_trade_contexts.pop(pair, None)
            self._persist_state(f"closed-trade:{trade_id}")
            if self.trade_sink is not None:
                try:
                    self.trade_sink(outbox_payload)
                except Exception:
                    pass
        return result

    def _record_close_context(
        self,
        *,
        pair: str,
        context: dict[str, Any],
        event_at_utc: str,
        fill_amount: Decimal,
        fill_price: Decimal,
        realized: Decimal,
        fee: Decimal,
        spread_cost: Decimal,
        adverse_cost: Decimal,
        exit_reason: str,
    ) -> dict[str, Any]:
        updated = dict(context)
        updated["closed_quantity"] = _s(
            _d(updated.get("closed_quantity", "0")) + fill_amount
        )
        updated["close_notional_jpy"] = _s(
            _d(updated.get("close_notional_jpy", "0"))
            + fill_amount * fill_price
        )
        updated["close_fees_jpy"] = _s(
            _d(updated.get("close_fees_jpy", "0")) + fee
        )
        updated["close_spread_cost_jpy"] = _s(
            _d(updated.get("close_spread_cost_jpy", "0"))
            + spread_cost
        )
        updated["close_adverse_cost_jpy"] = _s(
            _d(updated.get("close_adverse_cost_jpy", "0"))
            + adverse_cost
        )
        updated["realized_pnl_jpy"] = _s(
            _d(updated.get("realized_pnl_jpy", "0")) + realized
        )
        updated["last_closed_at_utc"] = event_at_utc
        updated["last_exit_reason"] = exit_reason
        self.state.open_trade_contexts[pair] = updated
        return updated

    def _record_open_context(
        self,
        *,
        pair: str,
        intent: dict[str, Any],
        position_side: str,
        event_at_utc: str,
        fill_amount: Decimal,
        fill_price: Decimal,
        notional: Decimal,
        fee: Decimal,
        spread_cost: Decimal,
        adverse_cost: Decimal,
    ) -> None:
        existing = self.state.open_trade_contexts.get(pair)
        if existing:
            quantity = _d(existing["quantity"]) + fill_amount
            total_notional = _d(existing["entry_notional_jpy"]) + notional
            existing.update(
                {
                    "quantity": _s(quantity),
                    "entry_notional_jpy": _s(total_notional),
                    "entry_price": _s(total_notional / quantity),
                    "open_fees_jpy": _s(
                        _d(existing["open_fees_jpy"]) + fee
                    ),
                    "open_spread_cost_jpy": _s(
                        _d(existing["open_spread_cost_jpy"]) + spread_cost
                    ),
                    "open_adverse_cost_jpy": _s(
                        _d(existing["open_adverse_cost_jpy"])
                        + adverse_cost
                    ),
                }
            )
            return
        raw_trade_id = (
            f"{intent.get('run_id', 'unknown')}|{intent['intent_id']}|"
            f"{pair}|{position_side}"
        )
        self.state.open_trade_contexts[pair] = {
            "trade_id": hashlib.sha256(raw_trade_id.encode()).hexdigest()[:24],
            "run_id": str(intent.get("run_id", "unknown")),
            "opened_at_utc": event_at_utc,
            "entry_price": _s(fill_price),
            "quantity": _s(fill_amount),
            "entry_notional_jpy": _s(notional),
            "open_fees_jpy": _s(fee),
            "open_spread_cost_jpy": _s(spread_cost),
            "open_adverse_cost_jpy": _s(adverse_cost),
            "interest_at_open_jpy": _s(
                self.state.interest_cost_by_pair.get(pair, Decimal("0"))
            ),
            "strategy": str(intent.get("strategy", "FAST_MICROSTRUCTURE")),
            "regime": str(intent.get("regime", "UNKNOWN")),
            "signal_reason": str(intent.get("signal_reason", "UNKNOWN")),
        }

    def _closed_trade_payload(
        self,
        *,
        pair: str,
        intent: dict[str, Any],
        context: dict[str, Any],
        position_side: str,
        event_at_utc: str,
        fill_amount: Decimal,
        fill_price: Decimal,
        realized: Decimal,
        close_fee: Decimal,
        close_spread: Decimal,
        close_adverse: Decimal,
    ) -> dict[str, Any]:
        opened = str(context.get("opened_at_utc", event_at_utc))
        try:
            holding_ms = max(
                0,
                int(
                    (
                        datetime.fromisoformat(event_at_utc)
                        - datetime.fromisoformat(opened)
                    ).total_seconds()
                    * 1000
                ),
            )
        except ValueError:
            holding_ms = 0
        entry_price = _d(context.get("entry_price", "0"))
        closed_quantity = _d(
            context.get("closed_quantity", fill_amount)
        )
        close_notional = _d(
            context.get(
                "close_notional_jpy", _s(fill_price * fill_amount)
            )
        )
        weighted_exit_price = (
            close_notional / closed_quantity
            if closed_quantity > 0
            else fill_price
        )
        gross_pnl = (
            (weighted_exit_price - entry_price) * closed_quantity
            if position_side == "LONG"
            else (entry_price - weighted_exit_price) * closed_quantity
        )
        fees = _d(context.get("open_fees_jpy", "0")) + _d(
            context.get("close_fees_jpy", close_fee)
        )
        spread = (
            _d(context.get("open_spread_cost_jpy", "0"))
            + _d(context.get("close_spread_cost_jpy", close_spread))
        )
        adverse = (
            _d(context.get("open_adverse_cost_jpy", "0"))
            + _d(context.get("close_adverse_cost_jpy", close_adverse))
        )
        interest = self.state.interest_cost_by_pair.get(
            pair, Decimal("0")
        ) - _d(context.get("interest_at_open_jpy", "0"))
        return {
            "trade_id": str(
                context.get(
                    "trade_id",
                    hashlib.sha256(
                        f"{intent['intent_id']}|{pair}".encode()
                    ).hexdigest()[:24],
                )
            ),
            "run_id": str(
                context.get("run_id", intent.get("run_id", "unknown"))
            ),
            "paper_mode": "MARGIN" if self.allow_short else "SPOT",
            "pair": pair,
            "side": position_side,
            "opened_at_utc": opened,
            "closed_at_utc": event_at_utc,
            "entry_price": _s(entry_price),
            "exit_price": _s(weighted_exit_price),
            "quantity": _s(closed_quantity),
            "entry_notional_jpy": str(
                context.get(
                    "entry_notional_jpy", _s(entry_price * fill_amount)
                )
            ),
            "gross_pnl_jpy": _s(gross_pnl),
            "fees_jpy": _s(fees),
            "spread_cost_jpy": _s(spread),
            "adverse_cost_jpy": _s(adverse),
            "funding_interest_jpy": _s(interest),
            "net_pnl_jpy": _s(
                _d(context.get("realized_pnl_jpy", realized)) - interest
            ),
            "holding_ms": holding_ms,
            "exit_reason": str(
                context.get(
                    "last_exit_reason",
                    intent.get(
                        "signal_reason", intent.get("regime", "UNKNOWN")
                    ),
                )
            ),
            "strategy": str(
                context.get("strategy", "FAST_MICROSTRUCTURE")
            ),
            "regime": str(context.get("regime", "UNKNOWN")),
            "guardian": str(intent.get("guardian", "GREEN")),
            "authority": "NONE",
            "live_permission": False,
        }

    def _opening_notional_capacity(self) -> Decimal:
        equity_at_cost = self.state.cash_jpy
        gross_exposure = Decimal("0")
        for pair, amount in self.state.positions.items():
            average = self.state.average_costs.get(pair, Decimal("0"))
            equity_at_cost += amount * average
            gross_exposure += abs(amount) * average
        return max(
            Decimal("0"),
            max(Decimal("0"), equity_at_cost) * self.max_leverage
            - gross_exposure,
        )

    def _record_realized(
        self,
        pair: str,
        fallback_regime: str,
        realized: Decimal,
        new_amount: Decimal,
    ) -> None:
        self.state.realized_pnl_by_pair[pair] = (
            self.state.realized_pnl_by_pair.get(pair, Decimal("0")) + realized
        )
        position_regime = self.state.position_regimes.get(
            pair, fallback_regime
        )
        self.state.realized_pnl_by_regime[position_regime] = (
            self.state.realized_pnl_by_regime.get(
                position_regime, Decimal("0")
            )
            + realized
        )
        if realized >= 0:
            self.state.gross_profit_jpy += realized
        else:
            self.state.gross_loss_jpy += -realized
        if new_amount == 0:
            self.state.average_costs.pop(pair, None)
            self.state.position_regimes.pop(pair, None)
            self.state.round_trips += 1

    @staticmethod
    def _consume_levels(
        levels: list[list[Any]], requested_amount: Decimal
    ) -> tuple[Decimal, Decimal, Decimal]:
        remaining = requested_amount
        cost = Decimal("0")
        filled = Decimal("0")
        best = _d(levels[0][0]) if levels else Decimal("0")
        for raw_price, raw_amount in levels:
            price = _d(raw_price)
            available = _d(raw_amount)
            take = min(remaining, available)
            filled += take
            cost += take * price
            remaining -= take
            if remaining <= 0:
                break
        return filled, cost / filled if filled else Decimal("0"), best

    def accrue_interest(
        self,
        daily_rates: dict[str, tuple[Decimal, Decimal]],
        *,
        elapsed_sec: float,
        cause_id: str,
    ) -> Decimal:
        if elapsed_sec <= 0:
            return Decimal("0")
        cost = Decimal("0")
        seconds = Decimal(str(elapsed_sec))
        for pair, amount in self.state.positions.items():
            if amount == 0:
                continue
            long_rate, short_rate = daily_rates.get(
                pair, (Decimal("0"), Decimal("0"))
            )
            rate = long_rate if amount > 0 else short_rate
            average = self.state.average_costs.get(pair, Decimal("0"))
            cost += abs(amount) * average * rate * seconds / Decimal("86400")
            pair_cost = (
                abs(amount) * average * rate * seconds / Decimal("86400")
            )
            self.state.interest_cost_by_pair[pair] = (
                self.state.interest_cost_by_pair.get(pair, Decimal("0"))
                + pair_cost
            )
        if cost > 0:
            self.state.cash_jpy -= cost
            self.state.interest_cost_jpy += cost
            self._persist_state(f"interest:{cause_id}")
        return cost

    def margin_snapshot(
        self,
        bids: dict[str, Decimal],
        asks: dict[str, Decimal] | None = None,
    ) -> dict[str, Any]:
        asks = asks or bids
        position_value = Decimal("0")
        gross_exposure = Decimal("0")
        unrealized_by_pair: dict[str, Decimal] = {}
        for pair, amount in self.state.positions.items():
            if amount == 0:
                continue
            mark = (
                bids.get(pair, Decimal("0"))
                if amount > 0
                else asks.get(pair, bids.get(pair, Decimal("0")))
            )
            position_value += amount * mark
            gross_exposure += abs(amount) * mark
            average = self.state.average_costs.get(pair, Decimal("0"))
            unrealized_by_pair[pair] = (
                amount * (mark - average)
                if amount > 0
                else abs(amount) * (average - mark)
            )
        equity = self.state.cash_jpy + position_value
        margin_ratio = (
            equity / gross_exposure
            if gross_exposure > 0
            else None
        )
        status = "NORMAL"
        if margin_ratio is not None:
            if margin_ratio <= self.maintenance_margin_ratio:
                status = "MODELED_LOSSCUT"
            elif margin_ratio < self.margin_call_ratio:
                status = "MODELED_MARGIN_CALL"
        return {
            "equity": equity,
            "gross_exposure": gross_exposure,
            "effective_leverage": (
                gross_exposure / equity
                if equity > 0
                else Decimal("Infinity")
            ),
            "margin_ratio": margin_ratio,
            "status": status,
            "unrealized_by_pair": unrealized_by_pair,
        }

    def mark_to_market(
        self,
        bids: dict[str, Decimal],
        asks: dict[str, Decimal] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.margin_snapshot(bids, asks)
        equity = snapshot["equity"]
        self.state.peak_equity_jpy = max(self.state.peak_equity_jpy, equity)
        self.state.max_drawdown_jpy = max(
            self.state.max_drawdown_jpy, self.state.peak_equity_jpy - equity
        )
        net_pnl = equity - self.state.initial_cash_jpy
        unrealized_by_pair = snapshot["unrealized_by_pair"]
        by_pair = dict(self.state.realized_pnl_by_pair)
        for pair, pnl in unrealized_by_pair.items():
            by_pair[pair] = by_pair.get(pair, Decimal("0")) + pnl
        by_regime = dict(self.state.realized_pnl_by_regime)
        for pair, pnl in unrealized_by_pair.items():
            regime = self.state.position_regimes.get(pair, "UNKNOWN")
            by_regime[regime] = by_regime.get(regime, Decimal("0")) + pnl
        gross_profit = self.state.gross_profit_jpy
        gross_loss = self.state.gross_loss_jpy
        profit_factor = (
            _s(gross_profit / gross_loss)
            if gross_loss > 0
            else None
        )
        metrics = {
            "initial_cash_jpy": _s(self.state.initial_cash_jpy),
            "equity_jpy": _s(equity),
            "net_pnl_jpy": _s(net_pnl),
            "max_drawdown_jpy": _s(self.state.max_drawdown_jpy),
            "profit_factor": profit_factor,
            "fill_count": self.state.fills,
            "completed_trade_count": self.state.round_trips,
            "trade_count": self.state.fills,
            "trade_count_semantics": (
                "DEPRECATED_FILL_EVENTS_USE_COMPLETED_TRADE_COUNT"
            ),
            "round_trip_count": self.state.round_trips,
            "turnover_jpy": _s(self.state.turnover_jpy),
            "gross_exposure_jpy": _s(snapshot["gross_exposure"]),
            "effective_leverage": (
                _s(snapshot["effective_leverage"])
                if snapshot["effective_leverage"].is_finite()
                else None
            ),
            "margin_ratio": (
                _s(snapshot["margin_ratio"])
                if snapshot["margin_ratio"] is not None
                else None
            ),
            "margin_status": snapshot["status"],
            "short_position_count": sum(
                amount < 0 for amount in self.state.positions.values()
            ),
            "open_position_count": sum(
                amount != 0 for amount in self.state.positions.values()
            ),
            "by_pair_pnl_jpy": {
                pair: _s(pnl) for pair, pnl in by_pair.items()
            },
            "by_regime_pnl_jpy": {
                regime: _s(pnl) for regime, pnl in by_regime.items()
            },
            "fees_jpy": _s(self.state.fees_jpy),
            "interest_cost_jpy": _s(self.state.interest_cost_jpy),
            "spread_cost_jpy": _s(self.state.spread_cost_jpy),
            "slippage_cost_jpy": _s(self.state.slippage_cost_jpy),
            "partial_fill_count": self.state.partial_fills,
            "partial_fill_ratio": (
                self.state.partial_fills / self.state.fills
                if self.state.fills
                else None
            ),
            "unfilled_order_count": self.state.unfilled_orders,
            "maker_fill_count": self.state.maker_fills,
            "taker_fill_count": self.state.taker_fills,
            "discipline_violations": self.state.discipline_violations,
            "margin_call_count": self.state.margin_calls,
            "forced_liquidation_count": self.state.forced_liquidations,
        }
        digest = hashlib.sha256(
            json_bytes(metrics)
        ).hexdigest()[:16]
        self.ledger.append(
            "PAPER_PNL",
            digest,
            metrics,
            dedupe_key=f"paper-pnl:{digest}:{metrics['trade_count']}",
        )
        self._persist_state(digest)
        return metrics

    def _persist_state(self, cause_id: str) -> None:
        payload = self.state.as_dict()
        payload["cause_id"] = cause_id
        digest = hashlib.sha256(
            json_bytes(payload)
        ).hexdigest()[:24]
        self.ledger.append(
            "PAPER_STATE",
            digest,
            payload,
            dedupe_key=f"paper-state:{digest}",
        )


def json_bytes(value: dict[str, Any]) -> bytes:
    import json

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
