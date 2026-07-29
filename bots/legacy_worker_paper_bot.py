"""Paper-only forward worker for archived TrendMA and PulseBreak families.

The worker deliberately exposes no live/broker client.  It receives only the
DOJO virtual broker and completed M1 bars.  ``BOT_ONLY`` preserves the frozen
family rule; ``AI_INVENTORY`` adds a frozen, model-authored, trim/protect-only
inventory policy and records every decision in a separate hash-chained ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_lab_provenance import OwnedBrokerView
from quant_rabbit.dojo_legacy_worker_comparison import AUTHORITY, POLICY_CONTRACT
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


PIP = 0.01


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class _State:
    def __init__(self) -> None:
        self.closes: deque[float] = deque(maxlen=1500)
        self.highs: deque[float] = deque(maxlen=100)
        self.lows: deque[float] = deque(maxlen=100)
        self.true_ranges: deque[float] = deque(maxlen=100)
        self.ema10: float | None = None
        self.ema20: float | None = None
        self.ema50: float | None = None
        self.trade_entry_atr: dict[str, float] = {}
        self.trade_risk: dict[str, float] = {}
        self.favorable: dict[str, float] = {}
        self.partial_done: set[str] = set()
        self.breakeven_done: set[str] = set()
        self.opposition_bars: dict[str, int] = {}


class Bot:
    def __init__(self, broker: VirtualBroker, cfg: dict | None = None):
        config = dict(cfg or json.loads(os.environ["DOJO_BOT_CONFIG"]))
        authority = config.pop("authority", None)
        if authority != AUTHORITY:
            raise ValueError("legacy Paper bot authority is invalid")
        self.family = str(config["family"])
        if self.family not in {"TrendMA", "PulseBreak"}:
            raise ValueError("unsupported archived family")
        self.arm = str(config["management_arm"])
        if self.arm not in {"BOT_ONLY", "AI_INVENTORY"}:
            raise ValueError("unsupported management arm")
        self.owner_id = str(config["strategy_owner_id"])
        self.pairs = list(config.get("pairs") or ["USD_JPY"])
        if self.pairs != ["USD_JPY"]:
            raise ValueError("archived worker forward room is USD_JPY only")
        self.risk_fraction = float(config.get("risk_fraction", 0.01))
        self.tp_pips = float(config["tp_pips"])
        self.sl_pips = float(config["sl_pips"])
        self.ceiling_bars = int(config["ceiling_bars"])
        if not (0 < self.risk_fraction <= 0.02 and self.tp_pips > 0 and self.sl_pips > 0):
            raise ValueError("invalid Paper risk geometry")
        self.broker = OwnedBrokerView(
            broker,
            self.owner_id,
            max_concurrent_per_pair=1,
            global_max_concurrent=1,
        )
        self.state = {pair: _State() for pair in self.pairs}
        self.opened_epoch: dict[str, int] = {}
        self.policy: dict[str, Any] | None = None
        self.decision_ledger: Path | None = None
        self.decision_tip = "0" * 64
        if self.arm == "AI_INVENTORY":
            policy_path = Path(str(config["ai_policy_path"])).resolve()
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            if policy.get("contract") != POLICY_CONTRACT or policy.get("authority") != AUTHORITY:
                raise ValueError("AI inventory policy is invalid")
            self.policy = dict(policy["parameters"])
            ledger_raw = os.environ.get("DOJO_AI_DECISION_LEDGER")
            if not ledger_raw:
                raise ValueError("AI room requires a separate decision ledger")
            self.decision_ledger = Path(ledger_raw).resolve()
            if self.decision_ledger.exists():
                raise ValueError("AI decision ledger must be create-once")
            self._record_ai(
                0,
                None,
                "ROOM_START",
                {
                    "policy_contract": policy["contract"],
                    "policy_sha256": _canonical_sha256(policy),
                    "paper_only": True,
                },
            )

    @staticmethod
    def _mid(bar: dict, key: str) -> float:
        return (float(bar[f"bid_{key}"]) + float(bar[f"ask_{key}"])) / 2.0

    def _update(self, state: _State, bar: dict) -> None:
        close = self._mid(bar, "c")
        high = self._mid(bar, "h")
        low = self._mid(bar, "l")
        previous = state.closes[-1] if state.closes else close
        state.closes.append(close)
        state.highs.append(high)
        state.lows.append(low)
        state.true_ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
        for period, name in ((10, "ema10"), (20, "ema20"), (50, "ema50")):
            current = getattr(state, name)
            alpha = 2.0 / (period + 1.0)
            setattr(state, name, close if current is None else current + alpha * (close - current))

    @staticmethod
    def _atr(state: _State) -> float:
        values = list(state.true_ranges)[-14:]
        return sum(values) / len(values) if values else 0.0

    def _record_ai(self, epoch: int, trade_id: str | None, action: str, detail: dict[str, Any]) -> None:
        if self.decision_ledger is None:
            return
        body = {
            "contract": "QR_DOJO_LEGACY_AI_PAPER_DECISION_V1",
            "epoch": int(epoch),
            "family": self.family,
            "room_owner_id": self.owner_id,
            "trade_id": trade_id,
            "action": action,
            "detail": detail,
            "previous_sha256": self.decision_tip,
            "authority": AUTHORITY,
        }
        record = {**body, "sha256": _canonical_sha256(body)}
        self.decision_ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.decision_ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.decision_tip = record["sha256"]

    def seed_bar(self, pair: str, bar: dict) -> None:
        state = self.state.get(pair)
        if state is not None:
            self._update(state, bar)

    def _signal(self, state: _State) -> str | None:
        if len(state.closes) < 50 or None in (state.ema10, state.ema20, state.ema50):
            return None
        close = state.closes[-1]
        drift15 = (close - state.closes[-16]) / PIP
        atr_pips = self._atr(state) / PIP
        if self.family == "TrendMA":
            gap = (state.ema10 - state.ema20) / PIP
            if gap >= 0.47 and drift15 > 0 and atr_pips >= 0.55:
                return "LONG"
            if gap <= -0.47 and drift15 < 0 and atr_pips >= 0.55:
                return "SHORT"
            return None
        momentum = close - state.ema20
        bias_pips = (state.ema20 - state.ema50) / PIP
        mean_move = sum(
            abs(a - b) / PIP
            for a, b in zip(list(state.closes)[-6:-1], list(state.closes)[-5:])
        ) / 5.0
        if momentum > 0 and bias_pips > 2.5 and atr_pips >= 0.45 and mean_move >= 0.3:
            return "LONG"
        if momentum < 0 and bias_pips < -2.5 and atr_pips >= 0.45 and mean_move >= 0.3:
            return "SHORT"
        return None

    def _manage_ai(self, pair: str, state: _State, bar: dict, epoch: int) -> None:
        assert self.policy is not None
        for trade_id in self.broker.active_trade_ids(pair=pair):
            position = self.broker.position(trade_id)
            if position is None:
                continue
            entry_atr = state.trade_entry_atr.get(trade_id)
            risk = state.trade_risk.get(trade_id)
            if not entry_atr or not risk:
                continue
            if position.side == "LONG":
                favorable = max(state.favorable.get(trade_id, position.entry_price), float(bar["bid_h"]))
                mark = float(bar["bid_c"])
                profit = mark - position.entry_price
                favorable_r = (favorable - position.entry_price) / risk
            else:
                favorable = min(state.favorable.get(trade_id, position.entry_price), float(bar["ask_l"]))
                mark = float(bar["ask_c"])
                profit = position.entry_price - mark
                favorable_r = (position.entry_price - favorable) / risk
            state.favorable[trade_id] = favorable

            if trade_id not in state.partial_done and favorable_r >= float(self.policy["partial_trigger_r"]):
                close_units = math.floor(position.units * float(self.policy["partial_fraction"]))
                if close_units > 0 and close_units < position.units:
                    try:
                        self.broker.close_trade(trade_id, units=close_units)
                        state.partial_done.add(trade_id)
                        self._record_ai(epoch, trade_id, "PARTIAL_CLOSE", {"units": close_units, "favorable_r": favorable_r})
                        position = self.broker.position(trade_id)
                        if position is None:
                            continue
                    except VirtualBrokerError:
                        pass

            candidate: float | None = None
            if trade_id not in state.breakeven_done and favorable_r >= float(self.policy["breakeven_trigger_r"]):
                candidate = position.entry_price
                state.breakeven_done.add(trade_id)
                self._record_ai(epoch, trade_id, "BREAKEVEN", {"favorable_r": favorable_r})
            if favorable_r >= float(self.policy["trailing_trigger_r"]):
                distance = max(self._atr(state) * float(self.policy["trailing_atr_multiple"]), PIP)
                trail = favorable - distance if position.side == "LONG" else favorable + distance
                if candidate is None or (position.side == "LONG" and trail > candidate) or (
                    position.side == "SHORT" and trail < candidate
                ):
                    candidate = trail
            if candidate is not None:
                tighter = position.sl_price is None or (
                    candidate > position.sl_price if position.side == "LONG" else candidate < position.sl_price
                )
                executable = position.current_price
                valid = executable is not None and (
                    candidate < executable if position.side == "LONG" else candidate > executable
                )
                if tighter and valid:
                    try:
                        self.broker.set_exit(trade_id, tp_price=position.tp_price, sl_price=round(candidate, 3))
                        self._record_ai(epoch, trade_id, "TRAIL_OR_BE_SET", {"stop": round(candidate, 3)})
                    except VirtualBrokerError:
                        pass

            closes = list(state.closes)
            fast = sum(closes[-3:]) / min(3, len(closes))
            slow = sum(closes[-10:]) / min(10, len(closes))
            opposed = (position.side == "LONG" and fast < slow) or (position.side == "SHORT" and fast > slow)
            state.opposition_bars[trade_id] = state.opposition_bars.get(trade_id, 0) + 1 if opposed else 0
            current_r = profit / risk
            if (
                state.opposition_bars[trade_id] >= int(self.policy["early_exit_opposition_bars"])
                and current_r < 0.25
            ):
                try:
                    self.broker.close_trade(trade_id)
                    self._record_ai(epoch, trade_id, "EARLY_EXIT", {"current_r": current_r})
                except VirtualBrokerError:
                    pass

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        state = self.state.get(pair)
        if state is None:
            return
        self._update(state, bar)
        active = set(self.broker.active_trade_ids(pair=pair))
        for trade_id in list(self.opened_epoch):
            if trade_id not in active:
                self.opened_epoch.pop(trade_id, None)
                state.trade_entry_atr.pop(trade_id, None)
                state.trade_risk.pop(trade_id, None)
                state.favorable.pop(trade_id, None)
                state.opposition_bars.pop(trade_id, None)
            elif epoch - self.opened_epoch[trade_id] >= self.ceiling_bars * 60:
                try:
                    self.broker.close_trade(trade_id)
                except VirtualBrokerError:
                    pass
        if self.arm == "AI_INVENTORY":
            self._manage_ai(pair, state, bar, epoch)
        if self.broker.active_trade_ids(pair=pair):
            return
        side = self._signal(state)
        if side is None:
            return
        atr = self._atr(state)
        size_multiple = 1.0
        if self.arm == "AI_INVENTORY":
            assert self.policy is not None
            lookback = int(self.policy["direction_lookback_bars"])
            closes = list(state.closes)
            drift = (closes[-1] - closes[-1 - min(lookback, len(closes) - 1)]) / PIP
            blocked = (side == "LONG" and drift < -float(self.policy["direction_block_pips"])) or (
                side == "SHORT" and drift > float(self.policy["direction_block_pips"])
            )
            self._record_ai(epoch, None, "ENTRY_DIRECTION_CHECK", {"side": side, "drift_pips": drift, "blocked": blocked})
            if blocked:
                return
            if atr / PIP >= float(self.policy["high_volatility_atr_pips"]):
                size_multiple = float(self.policy["high_volatility_size_multiple"])
        try:
            entry_price = self.broker.executable_market_entry_price(pair, side)
            equity = float(self.broker.account()["equity_jpy"])
        except VirtualBrokerError:
            return
        risk_fraction = self.risk_fraction if self.arm == "BOT_ONLY" else float(self.policy["risk_fraction"])
        units = math.floor(equity * risk_fraction / (self.sl_pips * PIP) * size_multiple)
        if units <= 0:
            return
        try:
            trade_id = self.broker.market_order(
                pair,
                side,
                units,
                tp_pips=self.tp_pips,
                sl_pips=self.sl_pips,
            )
        except VirtualBrokerError:
            return
        self.opened_epoch[trade_id] = epoch
        state.trade_entry_atr[trade_id] = atr
        state.trade_risk[trade_id] = self.sl_pips * PIP
        state.favorable[trade_id] = entry_price
        if self.arm == "AI_INVENTORY":
            self._record_ai(epoch, trade_id, "DYNAMIC_LOT_AND_ENTRY", {"side": side, "units": units, "size_multiple": size_multiple})
