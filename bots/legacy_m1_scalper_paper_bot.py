"""Independent Paper-only Bot/AI pair for the frozen M1Scalper signal."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_lab_provenance import OwnedBrokerView
from quant_rabbit.dojo_legacy_m1_signal import CausalM1Signal
from quant_rabbit.dojo_legacy_worker_comparison import AUTHORITY
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


PIP = 0.01
POLICY_CONTRACT = "QR_DOJO_LEGACY_M1_AI_INVENTORY_POLICY_V1"
DECISION_CONTRACT = "QR_DOJO_LEGACY_M1_PAPER_DECISION_V1"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class Bot:
    """Receives completed M1 bars and can mutate only its virtual broker view."""

    def __init__(self, broker: VirtualBroker, cfg: dict | None = None):
        config = dict(cfg or json.loads(os.environ["DOJO_BOT_CONFIG"]))
        if config.get("authority") != AUTHORITY:
            raise ValueError("M1 Paper authority is invalid")
        self.arm = str(config.get("management_arm"))
        if self.arm not in {"BOT_ONLY", "AI_INVENTORY"}:
            raise ValueError("M1 Paper arm is invalid")
        self.owner_id = str(config["strategy_owner_id"])
        self.operation_id = str(config["operation_id"])
        if not self.operation_id.startswith("dojo-m1scalper-paper:"):
            raise ValueError("M1 Paper operation_id is not scoped")
        self.fixed_units = int(config["fixed_units"])
        if self.fixed_units <= 0 or self.fixed_units > 1000:
            raise ValueError("M1 Paper fixed units must not exceed legacy 1000")
        self.ceiling_bars = int(config.get("ceiling_bars", 10))
        if not 1 <= self.ceiling_bars <= 30:
            raise ValueError("M1 Paper ceiling must be in [1, 30] bars")
        if list(config.get("pairs") or []) != ["USD_JPY"]:
            raise ValueError("M1 Paper is USD_JPY only")

        self.broker = OwnedBrokerView(
            broker,
            self.owner_id,
            max_concurrent_per_pair=1,
            global_max_concurrent=1,
        )
        self.signal = CausalM1Signal()
        self.policy: dict[str, Any] | None = None
        if self.arm == "AI_INVENTORY":
            policy = json.loads(
                Path(str(config["ai_policy_path"])).resolve().read_text(
                    encoding="utf-8"
                )
            )
            if (
                policy.get("contract") != POLICY_CONTRACT
                or policy.get("authority") != AUTHORITY
            ):
                raise ValueError("M1 AI policy contract/authority is invalid")
            params = dict(policy.get("parameters") or {})
            required = {
                "allowed_utc_hours",
                "allowed_sides",
                "cooldown_seconds",
                "max_concurrent",
                "fixed_units_cap",
                "high_volatility_atr_pips",
                "high_volatility_size_multiple",
                "breakeven_trigger_r",
                "partial_trigger_r",
                "partial_fraction",
                "trailing_trigger_r",
                "trailing_atr_multiple",
                "early_exit_opposition_bars",
            }
            if set(params) != required:
                raise ValueError("M1 AI policy schema is invalid")
            if (
                int(params["max_concurrent"]) != 1
                or int(params["fixed_units_cap"]) > self.fixed_units
                or not 0 < float(params["high_volatility_size_multiple"]) <= 1
            ):
                raise ValueError("M1 AI policy may trim but never increase risk")
            self.policy = params

        ledger_raw = os.environ.get("DOJO_M1_DECISION_LEDGER")
        if not ledger_raw:
            raise ValueError("M1 Paper requires an independent decision ledger")
        self.ledger = Path(ledger_raw).resolve()
        if self.ledger.exists():
            raise ValueError("M1 decision ledger must be create-once")
        self.tip = "0" * 64
        self.sequence = 0
        self.pending_order_epoch: dict[str, int] = {}
        self.pending_signal: dict[str, dict[str, Any]] = {}
        self.opened_epoch: dict[str, int] = {}
        self.trade_risk: dict[str, float] = {}
        self.favorable: dict[str, float] = {}
        self.partial_done: set[str] = set()
        self.breakeven_done: set[str] = set()
        self.opposition_bars: dict[str, int] = {}
        self.last_submission_epoch: int | None = None
        self._record(
            0,
            "ROOM_START",
            {
                "arm": self.arm,
                "signal_manifest": self.signal.manifest(),
                "fixed_units": self.fixed_units,
                "lot_increase_allowed": False,
                "paper_only": True,
            },
        )

    def _record(
        self,
        epoch: int,
        action: str,
        detail: dict[str, Any],
        *,
        trade_id: str | None = None,
    ) -> None:
        self.sequence += 1
        body = {
            "contract": DECISION_CONTRACT,
            "operation_id": f"{self.operation_id}:{self.sequence:08d}",
            "room_operation_id": self.operation_id,
            "epoch": int(epoch),
            "arm": self.arm,
            "owner_id": self.owner_id,
            "trade_id": trade_id,
            "action": action,
            "detail": detail,
            "previous_sha256": self.tip,
            "authority": AUTHORITY,
        }
        record = {**body, "sha256": _sha(body)}
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self.tip = record["sha256"]

    def seed_bar(self, pair: str, bar: dict) -> None:
        if pair == "USD_JPY":
            self.signal.seed_bar(bar)

    def _expire_orders(self, epoch: int) -> None:
        active = set(self.broker.active_order_ids(pair="USD_JPY"))
        for order_id in list(self.pending_order_epoch):
            if order_id not in active:
                self.pending_order_epoch.pop(order_id, None)
                continue
            signal = self.pending_signal.get(order_id, {})
            expiry = int(signal.get("limit_expiry_seconds") or 60)
            if epoch - self.pending_order_epoch[order_id] >= expiry:
                try:
                    self.broker.cancel_order(order_id)
                    self._record(epoch, "LIMIT_EXPIRED", {"order_id": order_id})
                except VirtualBrokerError:
                    pass
                self.pending_order_epoch.pop(order_id, None)
                self.pending_signal.pop(order_id, None)

    def _discover_fills(self, epoch: int) -> None:
        for trade_id in self.broker.active_trade_ids(pair="USD_JPY"):
            if trade_id in self.opened_epoch:
                continue
            position = self.broker.position(trade_id)
            if position is None:
                continue
            self.opened_epoch[trade_id] = epoch
            candidate = next(iter(self.pending_signal.values()), {})
            sl_pips = float(candidate.get("sl_pips") or 9.0)
            self.trade_risk[trade_id] = sl_pips * PIP
            self.favorable[trade_id] = position.entry_price
            self._record(
                epoch,
                "PAPER_FILL_OBSERVED",
                {"side": position.side, "units": position.units},
                trade_id=trade_id,
            )
        active_orders = set(self.broker.active_order_ids(pair="USD_JPY"))
        for order_id in list(self.pending_signal):
            if order_id not in active_orders:
                self.pending_signal.pop(order_id, None)
                self.pending_order_epoch.pop(order_id, None)

    def _clean_closed(self) -> None:
        active = set(self.broker.active_trade_ids(pair="USD_JPY"))
        for trade_id in list(self.opened_epoch):
            if trade_id in active:
                continue
            self.opened_epoch.pop(trade_id, None)
            self.trade_risk.pop(trade_id, None)
            self.favorable.pop(trade_id, None)
            self.opposition_bars.pop(trade_id, None)
            self.partial_done.discard(trade_id)
            self.breakeven_done.discard(trade_id)

    def _manage_ai(self, bar: dict, epoch: int) -> None:
        assert self.policy is not None
        closes = self.signal.closes
        for trade_id in self.broker.active_trade_ids(pair="USD_JPY"):
            position = self.broker.position(trade_id)
            risk = self.trade_risk.get(trade_id)
            if position is None or not risk:
                continue
            if position.side == "LONG":
                favorable = max(
                    self.favorable.get(trade_id, position.entry_price),
                    float(bar["bid_h"]),
                )
                profit = float(bar["bid_c"]) - position.entry_price
                favorable_r = (favorable - position.entry_price) / risk
            else:
                favorable = min(
                    self.favorable.get(trade_id, position.entry_price),
                    float(bar["ask_l"]),
                )
                profit = position.entry_price - float(bar["ask_c"])
                favorable_r = (position.entry_price - favorable) / risk
            self.favorable[trade_id] = favorable

            if (
                trade_id not in self.partial_done
                and favorable_r >= float(self.policy["partial_trigger_r"])
            ):
                units = math.floor(
                    position.units * float(self.policy["partial_fraction"])
                )
                if 0 < units < position.units:
                    try:
                        self.broker.close_trade(trade_id, units=units)
                        self.partial_done.add(trade_id)
                        self._record(
                            epoch,
                            "AI_PARTIAL_CLOSE",
                            {"units": units, "favorable_r": favorable_r},
                            trade_id=trade_id,
                        )
                        position = self.broker.position(trade_id)
                        if position is None:
                            continue
                    except VirtualBrokerError:
                        pass

            candidate: float | None = None
            if (
                trade_id not in self.breakeven_done
                and favorable_r >= float(self.policy["breakeven_trigger_r"])
            ):
                candidate = position.entry_price
                self.breakeven_done.add(trade_id)
            if favorable_r >= float(self.policy["trailing_trigger_r"]):
                distance = max(
                    self.signal.latest_atr_pips
                    * PIP
                    * float(self.policy["trailing_atr_multiple"]),
                    PIP,
                )
                trail = (
                    favorable - distance
                    if position.side == "LONG"
                    else favorable + distance
                )
                if candidate is None or (
                    position.side == "LONG" and trail > candidate
                ) or (position.side == "SHORT" and trail < candidate):
                    candidate = trail
            if candidate is not None:
                tighter = position.sl_price is None or (
                    candidate > position.sl_price
                    if position.side == "LONG"
                    else candidate < position.sl_price
                )
                executable = position.current_price
                valid = executable is not None and (
                    candidate < executable
                    if position.side == "LONG"
                    else candidate > executable
                )
                if tighter and valid:
                    try:
                        self.broker.set_exit(
                            trade_id,
                            tp_price=position.tp_price,
                            sl_price=round(candidate, 3),
                        )
                        self._record(
                            epoch,
                            "AI_TRAIL_OR_BREAKEVEN",
                            {"stop": round(candidate, 3)},
                            trade_id=trade_id,
                        )
                    except VirtualBrokerError:
                        pass

            fast = sum(closes[-3:]) / min(3, len(closes))
            slow = sum(closes[-10:]) / min(10, len(closes))
            opposed = (position.side == "LONG" and fast < slow) or (
                position.side == "SHORT" and fast > slow
            )
            self.opposition_bars[trade_id] = (
                self.opposition_bars.get(trade_id, 0) + 1 if opposed else 0
            )
            if (
                self.opposition_bars[trade_id]
                >= int(self.policy["early_exit_opposition_bars"])
                and profit / risk < 0.25
            ):
                try:
                    self.broker.close_trade(trade_id)
                    self._record(
                        epoch,
                        "AI_EARLY_EXIT",
                        {"current_r": profit / risk},
                        trade_id=trade_id,
                    )
                except VirtualBrokerError:
                    pass

    def _submit(self, signal: dict[str, Any], epoch: int) -> None:
        side = "LONG" if signal["action"] == "OPEN_LONG" else "SHORT"
        units = self.fixed_units
        if self.arm == "AI_INVENTORY":
            assert self.policy is not None
            allowed = (
                int(epoch // 3600 % 24) in set(self.policy["allowed_utc_hours"])
                and side in set(self.policy["allowed_sides"])
            )
            cooldown_ok = (
                self.last_submission_epoch is None
                or epoch - self.last_submission_epoch
                >= int(self.policy["cooldown_seconds"])
            )
            if self.signal.latest_atr_pips >= float(
                self.policy["high_volatility_atr_pips"]
            ):
                units = math.floor(
                    units * float(self.policy["high_volatility_size_multiple"])
                )
            self._record(
                epoch,
                "AI_ENTRY_DECISION",
                {
                    "side": side,
                    "utc_hour": int(epoch // 3600 % 24),
                    "session_direction_allowed": allowed,
                    "cooldown_allowed": cooldown_ok,
                    "atr_pips": self.signal.latest_atr_pips,
                    "units": units,
                    "lot_increase": False,
                },
            )
            if not allowed or not cooldown_ok:
                return
        if units <= 0 or units > self.fixed_units:
            return
        try:
            if signal.get("entry_type") == "limit":
                order_id = self.broker.limit_order(
                    "USD_JPY",
                    side,
                    units,
                    float(signal["entry_price"]),
                    tp_pips=float(signal["tp_pips"]),
                    sl_pips=float(signal["sl_pips"]),
                )
                self.pending_order_epoch[order_id] = epoch
                self.pending_signal[order_id] = dict(signal)
                target_id = order_id
                action = "PAPER_LIMIT_SUBMITTED"
            else:
                target_id = self.broker.market_order(
                    "USD_JPY",
                    side,
                    units,
                    tp_pips=float(signal["tp_pips"]),
                    sl_pips=float(signal["sl_pips"]),
                )
                self.trade_risk[target_id] = float(signal["sl_pips"]) * PIP
                self.opened_epoch[target_id] = epoch
                position = self.broker.position(target_id)
                if position is not None:
                    self.favorable[target_id] = position.entry_price
                action = "PAPER_MARKET_SUBMITTED"
        except VirtualBrokerError as exc:
            self._record(
                epoch, "PAPER_SUBMISSION_REJECTED", {"reason": str(exc)[:160]}
            )
            return
        self.last_submission_epoch = epoch
        self._record(
            epoch,
            action,
            {
                "target_id": target_id,
                "side": side,
                "units": units,
                "tp_pips": signal["tp_pips"],
                "sl_pips": signal["sl_pips"],
            },
        )

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        if pair != "USD_JPY":
            return
        signal = self.signal.on_bar_closed(bar)
        self._expire_orders(epoch)
        self._discover_fills(epoch)
        self._clean_closed()
        if self.arm == "AI_INVENTORY":
            self._manage_ai(bar, epoch)
        for trade_id, opened in list(self.opened_epoch.items()):
            if (
                epoch - opened >= self.ceiling_bars * 60
                and self.broker.position(trade_id) is not None
            ):
                try:
                    self.broker.close_trade(trade_id)
                    self._record(
                        epoch,
                        "TIME_CEILING_EXIT",
                        {"ceiling_bars": self.ceiling_bars},
                        trade_id=trade_id,
                    )
                except VirtualBrokerError:
                    pass
        if (
            signal is None
            or self.broker.active_trade_ids(pair=pair)
            or self.broker.active_order_ids(pair=pair)
        ):
            return
        self._record(
            epoch,
            "BASE_SIGNAL",
            {
                key: signal.get(key)
                for key in (
                    "action",
                    "entry_type",
                    "entry_price",
                    "tp_pips",
                    "sl_pips",
                    "confidence",
                    "tag",
                )
            },
        )
        self._submit(signal, epoch)
