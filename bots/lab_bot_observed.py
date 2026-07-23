"""Observed DOJO lab bot for future completed-M1 diagnostic rooms.

Trading logic is inherited unchanged from ``lab_bot.py``.  This wrapper adds
one hash-chained decision record after every completed M1 so a no-entry result
has an explicit, prospectively recorded reason.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_PATH = Path(__file__).with_name("lab_bot.py")
_SPEC = importlib.util.spec_from_file_location(
    "quant_rabbit_dojo_lab_bot_observed_base",
    BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load base lab bot: {BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)


DECISION_CONTRACT = "QR_DOJO_BOT_DECISION_OBSERVATION_V1"


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class Bot(_BASE.Bot):
    """Unchanged strategy with explicit gate telemetry."""

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        before_positions = set(self.broker.positions)
        before_orders = set(self.broker.orders)
        super().on_bar_closed(pair, bar, epoch)

        state = self.state.get(pair)
        if state is None:
            return
        pip = _BASE._pip(pair)
        trend = self._trend(state)
        efficiency = self._efficiency_6h(state)
        atr_pips = state.atr / pip if state.atr is not None else None
        mid_open = (float(bar["bid_o"]) + float(bar["ask_o"])) / 2.0
        mid_high = (float(bar["bid_h"]) + float(bar["ask_h"])) / 2.0
        mid_low = (float(bar["bid_l"]) + float(bar["ask_l"])) / 2.0
        mid_close = (float(bar["bid_c"]) + float(bar["ask_c"])) / 2.0
        range_pips = (mid_high - mid_low) / pip
        spread_pips = (
            float(bar["ask_c"]) - float(bar["bid_c"])
        ) / pip
        tp_pips = (
            float(self.tp_atr) * float(atr_pips)
            if self.tp_atr is not None and atr_pips is not None
            else float(self.tp_pips)
        )
        owned_positions = [
            position
            for position in self.broker.positions.values()
            if position.pair == pair
            and position.strategy_tag == self.strategy_tag
        ]
        owned_orders = [
            order
            for order in self.broker.orders.values()
            if order.pair == pair
            and order.strategy_tag == self.strategy_tag
        ]
        total_owned = sum(
            len(pair_state.my_trades)
            for pair_state in self.state.values()
        )
        capacity_full = (
            len(state.my_trades) >= self.max_concurrent
            or total_owned >= self.global_max
        )
        new_positions = sorted(set(self.broker.positions) - before_positions)
        new_orders = sorted(set(self.broker.orders) - before_orders)

        reason = "ORDER_NOT_CREATED"
        supervision = "STOP"
        if new_positions:
            reason = "POSITION_OPENED"
            supervision = "GO"
        elif owned_orders:
            reason = "PASSIVE_ORDER_ACTIVE"
            supervision = "GO"
        elif trend is None:
            reason = "TREND_WARMUP"
        elif atr_pips is None:
            reason = "ATR_WARMUP"
        elif atr_pips < self.atr_floor:
            reason = "ATR_BELOW_FLOOR"
        elif efficiency is None:
            reason = "EFFICIENCY_UNAVAILABLE"
        elif self._dd_tripped:
            reason = "DAILY_DRAWDOWN_BLOCK"
        elif capacity_full:
            reason = "CAPACITY_FULL"
        elif tp_pips <= 0 or spread_pips > tp_pips * 0.35:
            reason = "COST_HABITAT_BLOCK"
        elif self.signal == "range_fade_limit" and efficiency > self.eff_max:
            reason = "EFFICIENCY_ABOVE_MAX"
        elif (
            self.signal == "spike_fade"
            and state.atr is not None
            and (mid_high - mid_low) < 2.5 * state.atr
        ):
            reason = "SPIKE_RANGE_BELOW_THRESHOLD"

        body = {
            "contract": DECISION_CONTRACT,
            "pair": pair,
            "strategy_tag": self.strategy_tag,
            "signal": self.signal,
            "bar_start_utc": datetime.fromtimestamp(
                int(epoch),
                timezone.utc,
            ).isoformat(),
            "bar_end_utc": datetime.fromtimestamp(
                int(epoch) + 60,
                timezone.utc,
            ).isoformat(),
            "inputs": {
                "mid_open": mid_open,
                "mid_high": mid_high,
                "mid_low": mid_low,
                "mid_close": mid_close,
                "bar_range_pips": range_pips,
                "spread_pips": spread_pips,
                "trend_24h": trend,
                "atr_pips": atr_pips,
                "atr_floor_pips": self.atr_floor,
                "efficiency_6h": efficiency,
                "efficiency_max": self.eff_max,
                "spike_range_atr_min": (
                    2.5 if self.signal == "spike_fade" else None
                ),
                "tp_pips": tp_pips,
            },
            "inventory": {
                "owned_positions": len(owned_positions),
                "owned_orders": len(owned_orders),
                "max_concurrent": self.max_concurrent,
                "global_owned_positions": total_owned,
                "global_max_concurrent": self.global_max,
            },
            "result": {
                "reason": reason,
                "supervision": supervision,
                "new_position_ids": new_positions,
                "new_order_ids": new_orders,
            },
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        self.broker._log(
            "BOT_DECISION",
            {
                **body,
                "decision_sha256": _canonical_sha256(body),
            },
        )


__all__ = ["Bot", "DECISION_CONTRACT"]
