"""DOJO bot template — copy this file, rename the class logic, run it with:

    python3 scripts/run-virtual-market-session.py --feed replay \
        --session-dir <dir> --bot-module bots/example_bot.py:Bot \
        --from 2026-01-05T00:00:00 --to 2026-02-01T00:00:00

Contract: the class takes the shared VirtualBroker; on_bar_closed fires
once per completed M1 bar per pair (both replay and live feeds).  Trade
through the broker API only — fills happen exclusively at real quotes,
everything is hash-chain logged, and there is no path to the real broker.

bar dict keys: epoch, bid_o/h/l/c, ask_o/h/l/c
broker API:    market_order(pair, side, units, tp_pips=, sl_pips=)
               limit_order(pair, side, units, price=, tp_pips=, sl_pips=)
               close_trade(trade_id, units=None)
               set_exit(trade_id, tp_price=, sl_price=)
               cancel_order(order_id) / account() / positions (dict)
"""

from __future__ import annotations

from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


class Bot:
    """Example: buy a 20-bar breakout with TP 10 / SL 6, one position."""

    def __init__(self, broker: VirtualBroker):
        self.broker = broker
        self.highs: list[float] = []
        self.my_trade: str | None = None

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        if pair != "USD_JPY":
            return
        mid_high = (bar["bid_h"] + bar["ask_h"]) / 2
        mid_close = (bar["bid_c"] + bar["ask_c"]) / 2
        prior_max = max(self.highs) if len(self.highs) >= 20 else None
        self.highs.append(mid_high)
        if len(self.highs) > 20:
            self.highs.pop(0)
        if self.my_trade is not None and self.my_trade not in self.broker.positions:
            self.my_trade = None
        if self.my_trade is not None or prior_max is None:
            return
        if mid_close > prior_max:
            try:
                acct = self.broker.account()
                units = max(acct["equity_jpy"], 0.0) * 2.0 / mid_close
                self.my_trade = self.broker.market_order(
                    "USD_JPY", "LONG", units, tp_pips=10, sl_pips=6)
            except VirtualBrokerError:
                pass
