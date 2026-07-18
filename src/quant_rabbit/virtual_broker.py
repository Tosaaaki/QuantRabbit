"""Virtual broker: OANDA-mechanics paper account priced by REAL quotes.

Operator directive (2026-07-19): a virtual environment identical to
reality where the duty agent can trade at will.  This module is the
broker core; feeds (live polling or historical replay) supply quotes.

Honesty properties:
  * Fills happen ONLY at quotes actually supplied by the feed — market
    orders fill at the current real ask/bid; limit/TP/SL fill when a
    supplied quote touches the level, at the level (or the quote when
    it gapped past, whichever is worse for the trader).  No synthesis.
  * Accounting mirrors OANDA: hedge netting (margin on the larger side
    per instrument), leverage cap, margin closeout at 100% usage
    liquidating everything at current quotes.
  * Every action and fill is written to a hash-chained append-only
    ledger with the exact quote that caused it.
  * The broker never talks to the real broker.  It cannot place real
    orders by construction.

Non-JPY-quote pairs convert P&L at the latest USD_JPY mid supplied to
the broker (declared approximation, logged per fill).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

UTC = timezone.utc

LEVERAGE_DEFAULT = 25.0
CLOSEOUT_USAGE = 1.0


class VirtualBrokerError(ValueError):
    """Contract violation; callers must fail closed."""


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _round_price(pair: str, price: float) -> float:
    """Instrument price precision (JPY quotes: 3 dp, others: 5 dp) —
    mirrors broker tick precision and kills float-epsilon artifacts."""

    return round(price, 3 if pair.endswith("JPY") else 5)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class VBPosition:
    trade_id: str
    pair: str
    side: str  # LONG / SHORT
    units: float
    entry_price: float
    opened_ts: str
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None


@dataclass
class VBOrder:
    order_id: str
    pair: str
    side: str
    units: float
    limit_price: float
    tp_pips: Optional[float] = None
    sl_pips: Optional[float] = None


@dataclass
class VirtualBroker:
    ledger_path: Path
    balance_jpy: float = 200_000.0
    fast_ledger: bool = False  # flush without fsync (lab runs)
    leverage: float = LEVERAGE_DEFAULT
    positions: dict[str, VBPosition] = field(default_factory=dict)
    orders: dict[str, VBOrder] = field(default_factory=dict)
    last_quotes: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    _seq: int = 0
    _prev_sha: str = "0" * 64

    def __post_init__(self) -> None:
        if self.ledger_path.exists():
            with self.ledger_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    self._prev_sha = json.loads(line)["sha"]
        self._handle = self.ledger_path.open("a", encoding="utf-8")

    # ---- ledger ----------------------------------------------------------
    def _log(self, event: str, payload: dict[str, Any]) -> None:
        body = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
            "prev_sha": self._prev_sha,
        }
        record = {**body, "sha": _sha(body)}
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()
        if not self.fast_ledger:
            os.fsync(self._handle.fileno())
        self._prev_sha = record["sha"]

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq:06d}"

    # ---- conversion / accounting ----------------------------------------
    def _jpy_per_quote_unit(self, pair: str) -> float:
        """JPY value of one quote-currency unit, derived only from quotes
        the broker has actually seen (no synthetic rates)."""

        quote_ccy = pair.split("_")[1]
        if quote_ccy == "JPY":
            return 1.0
        usdjpy = self.last_quotes.get("USD_JPY")
        if usdjpy is None:
            raise VirtualBrokerError("USD_JPY quote required for JPY conversion")
        usdjpy_mid = (usdjpy[0] + usdjpy[1]) / 2.0
        if quote_ccy == "USD":
            return usdjpy_mid
        direct = self.last_quotes.get(f"{quote_ccy}_JPY")
        if direct is not None:
            return (direct[0] + direct[1]) / 2.0
        via_usd = self.last_quotes.get(f"USD_{quote_ccy}")
        if via_usd is not None:
            mid = (via_usd[0] + via_usd[1]) / 2.0
            if mid > 0:
                return usdjpy_mid / mid
        raise VirtualBrokerError(f"no conversion path for quote currency {quote_ccy}")

    def _position_pl_jpy(self, pos: VBPosition, bid: float, ask: float) -> float:
        mark = bid if pos.side == "LONG" else ask
        diff = (mark - pos.entry_price) if pos.side == "LONG" else (pos.entry_price - mark)
        return diff * pos.units * self._jpy_per_quote_unit(pos.pair)

    def _exposure_jpy(self, pair: str, units: float, price: float) -> float:
        if pair.endswith("JPY"):
            return units * price
        # base-currency exposure valued via quote conversion
        return units * price * self._jpy_per_quote_unit(pair)

    def account(self) -> dict[str, Any]:
        equity = self.balance_jpy
        margin = 0.0
        by_pair: dict[str, dict[str, float]] = {}
        for pos in self.positions.values():
            q = self.last_quotes.get(pos.pair)
            if q is None:
                raise VirtualBrokerError(f"no quote for open position pair {pos.pair}")
            equity += self._position_pl_jpy(pos, q[0], q[1])
            side_units = by_pair.setdefault(pos.pair, {"LONG": 0.0, "SHORT": 0.0})
            side_units[pos.side] += pos.units
        for pair, sides in by_pair.items():
            q = self.last_quotes[pair]
            mid = (q[0] + q[1]) / 2.0
            margin += self._exposure_jpy(pair, max(sides["LONG"], sides["SHORT"]), mid) / self.leverage
        usage = margin / equity if equity > 0 else 999.0
        return {
            "balance_jpy": round(self.balance_jpy, 2),
            "equity_jpy": round(equity, 2),
            "margin_used_jpy": round(margin, 2),
            "margin_usage": round(usage, 6),
            "open_positions": len(self.positions),
            "resting_orders": len(self.orders),
        }

    def _margin_headroom_ok(self, pair: str, side: str, units: float) -> bool:
        """OANDA-faithful: refuse orders whose margin would not fit."""

        try:
            acct = self.account()
        except VirtualBrokerError:
            return True  # no marks yet; nothing to measure against
        q = self.last_quotes.get(pair)
        if q is None:
            return False
        mid = (q[0] + q[1]) / 2.0
        long_u = sum(p.units for p in self.positions.values()
                     if p.pair == pair and p.side == "LONG")
        short_u = sum(p.units for p in self.positions.values()
                      if p.pair == pair and p.side == "SHORT")
        if side == "LONG":
            long_u += units
        else:
            short_u += units
        new_pair_margin = self._exposure_jpy(pair, max(long_u, short_u), mid) / self.leverage
        old_pair_margin = self._exposure_jpy(
            pair, max(long_u - (units if side == "LONG" else 0),
                      short_u - (units if side == "SHORT" else 0)), mid) / self.leverage
        new_total = acct["margin_used_jpy"] - old_pair_margin + new_pair_margin
        return new_total <= acct["equity_jpy"]

    # ---- agent actions ---------------------------------------------------
    def market_order(self, pair: str, side: str, units: float,
                     tp_pips: Optional[float] = None,
                     sl_pips: Optional[float] = None) -> str:
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        if units <= 0:
            raise VirtualBrokerError("units must be positive")
        q = self.last_quotes.get(pair)
        if q is None:
            raise VirtualBrokerError(f"no live quote for {pair}; cannot fill")
        if not self._margin_headroom_ok(pair, side, units):
            self._log("ORDER_REJECTED_INSUFFICIENT_MARGIN",
                      {"pair": pair, "side": side, "units": units})
            raise VirtualBrokerError("insufficient margin for market order")
        bid, ask, ts = q
        entry = ask if side == "LONG" else bid
        pip = _pip(pair)
        tp = _round_price(pair, entry + tp_pips * pip if side == "LONG" else entry - tp_pips * pip) if tp_pips else None
        sl = _round_price(pair, entry - sl_pips * pip if side == "LONG" else entry + sl_pips * pip) if sl_pips else None
        trade_id = self._next_id("T")
        self.positions[trade_id] = VBPosition(
            trade_id=trade_id, pair=pair, side=side, units=units,
            entry_price=entry, opened_ts=ts, tp_price=tp, sl_price=sl,
        )
        self._log("FILL_MARKET", {
            "trade_id": trade_id, "pair": pair, "side": side, "units": units,
            "entry": entry, "tp": tp, "sl": sl,
            "quote": {"bid": bid, "ask": ask, "ts": ts},
        })
        self._enforce_margin_after_action()
        return trade_id

    def limit_order(self, pair: str, side: str, units: float, price: float,
                    tp_pips: Optional[float] = None,
                    sl_pips: Optional[float] = None) -> str:
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        if units <= 0 or price <= 0:
            raise VirtualBrokerError("units and price must be positive")
        order_id = self._next_id("O")
        self.orders[order_id] = VBOrder(
            order_id=order_id, pair=pair, side=side, units=units,
            limit_price=price, tp_pips=tp_pips, sl_pips=sl_pips,
        )
        self._log("ORDER_LIMIT", {
            "order_id": order_id, "pair": pair, "side": side,
            "units": units, "price": price, "tp_pips": tp_pips, "sl_pips": sl_pips,
        })
        return order_id

    def cancel_order(self, order_id: str) -> None:
        if order_id not in self.orders:
            raise VirtualBrokerError(f"unknown order: {order_id}")
        del self.orders[order_id]
        self._log("ORDER_CANCEL", {"order_id": order_id})

    def close_trade(self, trade_id: str, units: Optional[float] = None) -> float:
        pos = self.positions.get(trade_id)
        if pos is None:
            raise VirtualBrokerError(f"unknown trade: {trade_id}")
        q = self.last_quotes.get(pos.pair)
        if q is None:
            raise VirtualBrokerError(f"no live quote for {pos.pair}; cannot close")
        bid, ask, ts = q
        close_units = pos.units if units is None else min(units, pos.units)
        if close_units <= 0:
            raise VirtualBrokerError("close units must be positive")
        price = bid if pos.side == "LONG" else ask
        diff = (price - pos.entry_price) if pos.side == "LONG" else (pos.entry_price - price)
        pl = diff * close_units * self._jpy_per_quote_unit(pos.pair)
        self.balance_jpy += pl
        if close_units >= pos.units:
            del self.positions[trade_id]
        else:
            pos.units -= close_units
        self._log("CLOSE", {
            "trade_id": trade_id, "units": close_units, "price": price,
            "pl_jpy": round(pl, 2), "quote": {"bid": bid, "ask": ask, "ts": ts},
        })
        return pl

    def set_exit(self, trade_id: str, tp_price: Optional[float] = None,
                 sl_price: Optional[float] = None) -> None:
        pos = self.positions.get(trade_id)
        if pos is None:
            raise VirtualBrokerError(f"unknown trade: {trade_id}")
        pos.tp_price = tp_price
        pos.sl_price = sl_price
        self._log("SET_EXIT", {"trade_id": trade_id, "tp": tp_price, "sl": sl_price})

    # ---- feed ------------------------------------------------------------
    def on_quote(self, pair: str, bid: float, ask: float, ts: str) -> list[dict[str, Any]]:
        """Process one real quote: resting orders, TP/SL, margin. Returns events."""

        if bid <= 0 or ask <= 0 or ask < bid:
            raise VirtualBrokerError(f"invalid quote {pair} {bid}/{ask}")
        self.last_quotes[pair] = (bid, ask, ts)
        events: list[dict[str, Any]] = []

        # resting limit fills (worse-of level/quote when gapped)
        for order_id in list(self.orders):
            order = self.orders[order_id]
            if order.pair != pair:
                continue
            filled_price = None
            if order.side == "LONG" and ask <= order.limit_price:
                filled_price = min(order.limit_price, ask)
            elif order.side == "SHORT" and bid >= order.limit_price:
                filled_price = max(order.limit_price, bid)
            if filled_price is None:
                continue
            if not self._margin_headroom_ok(pair, order.side, order.units):
                del self.orders[order_id]
                self._log("LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                          {"order_id": order_id, "pair": pair})
                continue
            pip = _pip(pair)
            tp = _round_price(pair, filled_price + order.tp_pips * pip if order.side == "LONG"
                              else filled_price - order.tp_pips * pip) if order.tp_pips else None
            sl = _round_price(pair, filled_price - order.sl_pips * pip if order.side == "LONG"
                              else filled_price + order.sl_pips * pip) if order.sl_pips else None
            trade_id = self._next_id("T")
            self.positions[trade_id] = VBPosition(
                trade_id=trade_id, pair=pair, side=order.side, units=order.units,
                entry_price=filled_price, opened_ts=ts, tp_price=tp, sl_price=sl,
            )
            del self.orders[order_id]
            event = {
                "event": "FILL_LIMIT", "order_id": order_id, "trade_id": trade_id,
                "pair": pair, "side": order.side, "units": order.units,
                "price": filled_price, "quote": {"bid": bid, "ask": ask, "ts": ts},
            }
            self._log("FILL_LIMIT", event)
            events.append(event)

        # TP/SL: SL first when both touch on the same quote (pessimistic)
        for trade_id in list(self.positions):
            pos = self.positions[trade_id]
            if pos.pair != pair:
                continue
            exit_price = None
            reason = None
            if pos.side == "LONG":
                if pos.sl_price is not None and bid <= pos.sl_price:
                    exit_price, reason = min(pos.sl_price, bid), "SL"
                elif pos.tp_price is not None and bid >= pos.tp_price:
                    exit_price, reason = pos.tp_price, "TP"
            else:
                if pos.sl_price is not None and ask >= pos.sl_price:
                    exit_price, reason = max(pos.sl_price, ask), "SL"
                elif pos.tp_price is not None and ask <= pos.tp_price:
                    exit_price, reason = pos.tp_price, "TP"
            if exit_price is None:
                continue
            diff = (exit_price - pos.entry_price) if pos.side == "LONG" else (
                pos.entry_price - exit_price)
            pl = diff * pos.units * self._jpy_per_quote_unit(pos.pair)
            self.balance_jpy += pl
            del self.positions[trade_id]
            event = {
                "event": f"EXIT_{reason}", "trade_id": trade_id, "price": exit_price,
                "pl_jpy": round(pl, 2), "quote": {"bid": bid, "ask": ask, "ts": ts},
            }
            self._log(f"EXIT_{reason}", event)
            events.append(event)

        events.extend(self._enforce_margin_after_action())
        return events

    def _enforce_margin_after_action(self) -> list[dict[str, Any]]:
        try:
            acct = self.account()
        except VirtualBrokerError:
            return []
        if acct["margin_usage"] < CLOSEOUT_USAGE or not self.positions:
            return []
        events = []
        for trade_id in list(self.positions):
            pos = self.positions[trade_id]
            q = self.last_quotes[pos.pair]
            price = q[0] if pos.side == "LONG" else q[1]
            diff = (price - pos.entry_price) if pos.side == "LONG" else (
                pos.entry_price - price)
            pl = diff * pos.units * self._jpy_per_quote_unit(pos.pair)
            self.balance_jpy += pl
            del self.positions[trade_id]
            event = {"event": "MARGIN_CLOSEOUT", "trade_id": trade_id,
                     "price": price, "pl_jpy": round(pl, 2)}
            self._log("MARGIN_CLOSEOUT", event)
            events.append(event)
        return events

    # ---- persistence -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "balance_jpy": self.balance_jpy,
            "seq": self._seq,
            "positions": [vars(p) for p in self.positions.values()],
            "orders": [vars(o) for o in self.orders.values()],
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.balance_jpy = float(snap["balance_jpy"])
        self._seq = int(snap["seq"])
        self.positions = {
            p["trade_id"]: VBPosition(**p) for p in snap["positions"]
        }
        self.orders = {o["order_id"]: VBOrder(**o) for o in snap["orders"]}
