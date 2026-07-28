from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

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
    fees_jpy: Decimal = Decimal("0")
    spread_cost_jpy: Decimal = Decimal("0")
    slippage_cost_jpy: Decimal = Decimal("0")
    fills: int = 0
    partial_fills: int = 0
    unfilled_orders: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
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
            "fees_jpy": _s(self.fees_jpy),
            "spread_cost_jpy": _s(self.spread_cost_jpy),
            "slippage_cost_jpy": _s(self.slippage_cost_jpy),
            "fills": self.fills,
            "partial_fills": self.partial_fills,
            "unfilled_orders": self.unfilled_orders,
            "maker_fills": self.maker_fills,
            "taker_fills": self.taker_fills,
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
            fees_jpy=_d(raw["fees_jpy"]),
            spread_cost_jpy=_d(raw["spread_cost_jpy"]),
            slippage_cost_jpy=_d(raw["slippage_cost_jpy"]),
            fills=int(raw["fills"]),
            partial_fills=int(raw["partial_fills"]),
            unfilled_orders=int(raw["unfilled_orders"]),
            maker_fills=int(raw.get("maker_fills", 0)),
            taker_fills=int(raw.get("taker_fills", 0)),
            discipline_violations=int(raw["discipline_violations"]),
            peak_equity_jpy=_d(raw["peak_equity_jpy"]),
            max_drawdown_jpy=_d(raw["max_drawdown_jpy"]),
        )


class PaperEngine:
    """Conservative spot-long paper engine with no broker dependency."""

    def __init__(
        self,
        ledger: CryptoLedger,
        *,
        initial_cash_jpy: Decimal = Decimal("100000"),
        maker_fill_fraction: Decimal = Decimal("0.25"),
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
        self.maker_fill_fraction = maker_fill_fraction

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
        if intent.get("side") != "BUY":
            raise RuntimeError("Phase 1 paper engine permits spot BUY intents only")
        pair = str(intent["pair"])
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
            "side": "BUY",
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
            fill_amount, fill_price, best_price = self._consume_asks(
                depth.get("asks", []), requested_amount
            )
            fee_rate = taker_fee_rate
        else:
            bids = depth.get("bids", [])
            best_price = _d(bids[0][0]) if bids else Decimal("0")
            available = _d(bids[0][1]) if bids else Decimal("0")
            fill_amount = min(
                requested_amount * self.maker_fill_fraction,
                available * Decimal("0.10"),
            )
            fill_price = best_price
            fee_rate = maker_fee_rate
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
        if cash_delta > self.state.cash_jpy:
            fill_amount = max(
                Decimal("0"),
                self.state.cash_jpy / (fill_price * (Decimal("1") + fee_rate)),
            )
            notional = fill_amount * fill_price
            fee = notional * fee_rate
            cash_delta = notional + fee
        previous_amount = self.state.positions.get(pair, Decimal("0"))
        previous_cost = self.state.average_costs.get(pair, Decimal("0"))
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
        self.state.fees_jpy += fee
        spread_cost = max(Decimal("0"), (fill_price - best_price) * fill_amount)
        slippage_cost = spread_cost
        self.state.spread_cost_jpy += spread_cost
        self.state.slippage_cost_jpy += slippage_cost
        self.state.fills += 1
        if order_style == "PAPER_TAKER":
            self.state.taker_fills += 1
        else:
            self.state.maker_fills += 1
        remaining = max(Decimal("0"), requested_amount - fill_amount)
        status = "FILLED" if remaining == 0 else "PARTIALLY_FILLED"
        if remaining > 0:
            self.state.partial_fills += 1
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
        return result

    @staticmethod
    def _consume_asks(
        asks: list[list[Any]], requested_amount: Decimal
    ) -> tuple[Decimal, Decimal, Decimal]:
        remaining = requested_amount
        cost = Decimal("0")
        filled = Decimal("0")
        best = _d(asks[0][0]) if asks else Decimal("0")
        for raw_price, raw_amount in asks:
            price = _d(raw_price)
            available = _d(raw_amount)
            take = min(remaining, available)
            filled += take
            cost += take * price
            remaining -= take
            if remaining <= 0:
                break
        return filled, cost / filled if filled else Decimal("0"), best

    def mark_to_market(self, bids: dict[str, Decimal]) -> dict[str, Any]:
        position_value = sum(
            amount * bids.get(pair, Decimal("0"))
            for pair, amount in self.state.positions.items()
        )
        equity = self.state.cash_jpy + position_value
        self.state.peak_equity_jpy = max(self.state.peak_equity_jpy, equity)
        self.state.max_drawdown_jpy = max(
            self.state.max_drawdown_jpy, self.state.peak_equity_jpy - equity
        )
        net_pnl = equity - self.state.initial_cash_jpy
        by_pair = {
            pair: amount
            * (bids.get(pair, Decimal("0")) - self.state.average_costs[pair])
            for pair, amount in self.state.positions.items()
        }
        by_regime: dict[str, Decimal] = {}
        for pair, pnl in by_pair.items():
            regime = self.state.position_regimes.get(pair, "UNKNOWN")
            by_regime[regime] = by_regime.get(regime, Decimal("0")) + pnl
        gross_profit = sum(
            (pnl for pnl in by_pair.values() if pnl > 0), Decimal("0")
        )
        gross_loss = -sum(
            (pnl for pnl in by_pair.values() if pnl < 0), Decimal("0")
        )
        profit_factor = (
            _s(gross_profit / gross_loss)
            if gross_loss > 0
            else None
        )
        metrics = {
            "equity_jpy": _s(equity),
            "net_pnl_jpy": _s(net_pnl),
            "max_drawdown_jpy": _s(self.state.max_drawdown_jpy),
            "profit_factor": profit_factor,
            "trade_count": self.state.fills,
            "by_pair_pnl_jpy": {
                pair: _s(pnl) for pair, pnl in by_pair.items()
            },
            "by_regime_pnl_jpy": {
                regime: _s(pnl) for regime, pnl in by_regime.items()
            },
            "fees_jpy": _s(self.state.fees_jpy),
            "spread_cost_jpy": _s(self.state.spread_cost_jpy),
            "slippage_cost_jpy": _s(self.state.slippage_cost_jpy),
            "partial_fill_count": self.state.partial_fills,
            "unfilled_order_count": self.state.unfilled_orders,
            "maker_fill_count": self.state.maker_fills,
            "taker_fill_count": self.state.taker_fills,
            "discipline_violations": self.state.discipline_violations,
        }
        digest = hashlib.sha256(
            str(sorted((pair, str(price)) for pair, price in bids.items())).encode()
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
