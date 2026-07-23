"""Replay-only conditional inventory-release candidate.

This module deliberately wraps the unchanged level-fade entry worker instead
of changing ``bots/lab_bot.py``.  Active/fixed paper rooms therefore keep the
exact code already sealed in their session contracts.

The single candidate change is:

* after 30 minutes, release a position at the first completed M1 when the
  current 24-hour trend opposes the position, the current six-hour change
  agrees with that trend, and six-hour efficiency is at least 0.25.

The original TP, SL, 60-minute ceiling, entry geometry and sizing remain
unchanged.  The action is a virtual-broker event only; this module imports no
real broker client and has no live/order authority.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from quant_rabbit.virtual_broker import VirtualBrokerError


_BASELINE_PATH = Path(__file__).with_name("lab_bot.py")
_BASELINE_SPEC = importlib.util.spec_from_file_location(
    "dojo_inventory_release_baseline_bot", _BASELINE_PATH
)
if _BASELINE_SPEC is None or _BASELINE_SPEC.loader is None:
    raise RuntimeError(f"cannot load baseline bot: {_BASELINE_PATH}")
_BASELINE_MODULE = importlib.util.module_from_spec(_BASELINE_SPEC)
_BASELINE_SPEC.loader.exec_module(_BASELINE_MODULE)
BaselineBot = _BASELINE_MODULE.Bot
_pip = _BASELINE_MODULE._pip


class Bot(BaselineBot):
    """Level-fade bot with one deterministic replay-only exit condition."""

    def __init__(self, broker, cfg: dict | None = None):
        super().__init__(broker, cfg)
        if self.signal != "prev_day_extreme_fade":
            raise ValueError(
                "inventory release candidate is limited to prev_day_extreme_fade"
            )
        config = cfg or {}
        self.release_age_s = int(
            float(config.get("inventory_release_min_age_min", 30.0)) * 60
        )
        self.release_efficiency_min = float(
            config.get("inventory_release_efficiency_min", 0.25)
        )
        if self.release_age_s <= 0:
            raise ValueError("inventory release age must be positive")
        if not 0.0 <= self.release_efficiency_min <= 1.0:
            raise ValueError("inventory release efficiency must be in [0, 1]")

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        # The baseline owns all entries, TP/SL geometry and the original
        # ceiling.  With max_concurrent=1 it cannot place a replacement order
        # while the position remains open, so evaluating the one exit change
        # after the baseline update cannot alter entry selection.
        super().on_bar_closed(pair, bar, epoch)
        st = self.state.get(pair)
        if st is None:
            return
        trend = self._trend(st)
        efficiency = self._efficiency_6h(st)
        if (
            trend is None
            or efficiency is None
            or efficiency + 1e-12 < self.release_efficiency_min
            or len(st.closes) < 361
        ):
            return
        pip = _pip(pair)
        change_6h_pips = (st.closes[-1] - st.closes[-361]) / pip
        trend_change_agrees = (
            trend == "LONG" and change_6h_pips > 0.0
        ) or (
            trend == "SHORT" and change_6h_pips < 0.0
        )
        if not trend_change_agrees:
            return

        for trade_id, opened_epoch in list(st.my_trades.items()):
            position = self.broker.positions.get(trade_id)
            if (
                position is None
                or position.pair != pair
                or position.strategy_tag != self.strategy_tag
                or epoch - opened_epoch < self.release_age_s
                or position.side == trend
            ):
                continue
            self.broker._log(
                "INVENTORY_RELEASE_DECISION",
                {
                    "contract": "QR_DOJO_INVENTORY_RELEASE_DECISION_V1",
                    "paper_only": True,
                    "order_authority": "NONE",
                    "live_permission": False,
                    "trade_id": trade_id,
                    "pair": pair,
                    "strategy_tag": self.strategy_tag,
                    "position_side": position.side,
                    "trend_24h": trend,
                    "change_6h_pips": change_6h_pips,
                    "efficiency_6h": efficiency,
                    "age_seconds": epoch - opened_epoch,
                    "rule": "trend_conflict_inventory_release_after_30m",
                },
            )
            try:
                self.broker.close_trade(trade_id)
            except VirtualBrokerError:
                continue
            st.my_trades.pop(trade_id, None)
            self._owner.pop(trade_id, None)
