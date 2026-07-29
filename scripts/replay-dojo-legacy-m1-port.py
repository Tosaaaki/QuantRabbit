#!/usr/bin/env python3
"""Costed causal replay of the frozen M1Scalper Bot/AI Paper pair."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_legacy_m1_signal import CausalM1Signal, canonical_sha256
from quant_rabbit.dojo_legacy_worker_comparison import AUTHORITY


PIP = 0.01
CONTRACT = "QR_DOJO_LEGACY_M1_PORT_REPLAY_V1"


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for raw in payload["candles"]:
        if raw.get("complete") is not True:
            continue
        mid = raw["mid"]
        epoch = int(_time(str(raw["time"])).timestamp())
        output.append(
            {
                "epoch": epoch,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
            }
        )
    if not output or any(a["epoch"] >= b["epoch"] for a, b in zip(output, output[1:])):
        raise ValueError(f"empty or non-causal candle file: {path}")
    return output


def _signal_bar(row: dict[str, Any]) -> dict[str, Any]:
    half_spread = 0.005
    return {
        "epoch": row["epoch"],
        **{
            f"{side}_{key}": row[name] + (half_spread if side == "ask" else -half_spread)
            for side in ("bid", "ask")
            for key, name in zip("ohlc", ("open", "high", "low", "close"))
        },
    }


@dataclass
class Position:
    side: str
    entry: float
    tp: float
    stop: float
    units: int
    opened_index: int
    risk: float
    remaining: int
    favorable: float
    realized: float = 0.0
    partial_done: bool = False
    opposition: int = 0


@dataclass
class Arm:
    name: str
    policy: dict[str, Any] | None
    pending: dict[str, Any] | None = None
    position: Position | None = None
    last_submission_epoch: int | None = None
    trades: list[float] = field(default_factory=list)
    decisions: int = 0


def _exit_pnl(position: Position, price: float, units: int, cost_pips: float) -> float:
    direction = 1.0 if position.side == "LONG" else -1.0
    return (price - position.entry) * direction * units - cost_pips * PIP * units


def _close(arm: Arm, price: float, cost_pips: float) -> None:
    assert arm.position is not None
    pnl = arm.position.realized + _exit_pnl(
        arm.position, price, arm.position.remaining, cost_pips
    )
    arm.trades.append(pnl)
    arm.position = None


def _manage(
    arm: Arm,
    row: dict[str, Any],
    index: int,
    closes: list[float],
    atr_pips: float,
    cost_pips: float,
) -> None:
    position = arm.position
    if position is None or index <= position.opened_index:
        return
    if position.side == "LONG":
        if row["low"] <= position.stop:
            _close(arm, position.stop, cost_pips)
            return
        if row["high"] >= position.tp:
            _close(arm, position.tp, cost_pips)
            return
        position.favorable = max(position.favorable, row["high"])
        favorable_r = (position.favorable - position.entry) / position.risk
        current_r = (row["close"] - position.entry) / position.risk
    else:
        if row["high"] >= position.stop:
            _close(arm, position.stop, cost_pips)
            return
        if row["low"] <= position.tp:
            _close(arm, position.tp, cost_pips)
            return
        position.favorable = min(position.favorable, row["low"])
        favorable_r = (position.entry - position.favorable) / position.risk
        current_r = (position.entry - row["close"]) / position.risk

    if arm.policy is not None:
        policy = arm.policy
        if not position.partial_done and favorable_r >= float(policy["partial_trigger_r"]):
            units = math.floor(position.units * float(policy["partial_fraction"]))
            if 0 < units < position.remaining:
                price = position.entry + (
                    position.risk * float(policy["partial_trigger_r"])
                    * (1 if position.side == "LONG" else -1)
                )
                position.realized += _exit_pnl(position, price, units, cost_pips)
                position.remaining -= units
                position.partial_done = True
                arm.decisions += 1
        candidate = position.stop
        if favorable_r >= float(policy["breakeven_trigger_r"]):
            candidate = (
                max(candidate, position.entry)
                if position.side == "LONG"
                else min(candidate, position.entry)
            )
        if favorable_r >= float(policy["trailing_trigger_r"]):
            distance = max(
                atr_pips * PIP * float(policy["trailing_atr_multiple"]), PIP
            )
            trail = (
                position.favorable - distance
                if position.side == "LONG"
                else position.favorable + distance
            )
            candidate = (
                max(candidate, trail)
                if position.side == "LONG"
                else min(candidate, trail)
            )
        if candidate != position.stop and (
            (position.side == "LONG" and candidate < row["close"])
            or (position.side == "SHORT" and candidate > row["close"])
        ):
            position.stop = candidate
            arm.decisions += 1
        fast = sum(closes[-3:]) / min(3, len(closes))
        slow = sum(closes[-10:]) / min(10, len(closes))
        opposed = (position.side == "LONG" and fast < slow) or (
            position.side == "SHORT" and fast > slow
        )
        position.opposition = position.opposition + 1 if opposed else 0
        if (
            position.opposition >= int(policy["early_exit_opposition_bars"])
            and current_r < 0.25
        ):
            arm.decisions += 1
            _close(arm, row["close"], cost_pips)
            return
    if arm.position is not None and index - position.opened_index >= 10:
        _close(arm, row["close"], cost_pips)


def _try_fill(arm: Arm, row: dict[str, Any], index: int) -> None:
    pending = arm.pending
    if pending is None:
        return
    if index > pending["expires_index"]:
        arm.pending = None
        return
    side = pending["side"]
    entry = pending["entry"]
    touched = row["low"] <= entry if side == "LONG" else row["high"] >= entry
    if not touched:
        return
    arm.position = Position(
        side=side,
        entry=entry,
        tp=entry + pending["tp_pips"] * PIP * (1 if side == "LONG" else -1),
        stop=entry - pending["sl_pips"] * PIP * (1 if side == "LONG" else -1),
        units=pending["units"],
        opened_index=index,
        risk=pending["sl_pips"] * PIP,
        remaining=pending["units"],
        favorable=entry,
    )
    arm.pending = None


def _submit(
    arm: Arm,
    signal: dict[str, Any],
    row: dict[str, Any],
    index: int,
    atr_pips: float,
) -> None:
    side = "LONG" if signal["action"] == "OPEN_LONG" else "SHORT"
    units = 1000
    if arm.policy is not None:
        policy = arm.policy
        arm.decisions += 1
        allowed = (
            datetime.fromtimestamp(row["epoch"], tz=timezone.utc).hour
            in set(policy["allowed_utc_hours"])
            and side in set(policy["allowed_sides"])
        )
        cooldown = (
            arm.last_submission_epoch is None
            or row["epoch"] - arm.last_submission_epoch
            >= int(policy["cooldown_seconds"])
        )
        if not allowed or not cooldown:
            return
        if atr_pips >= float(policy["high_volatility_atr_pips"]):
            units = math.floor(
                units * float(policy["high_volatility_size_multiple"])
            )
    if signal.get("entry_type") == "limit":
        arm.pending = {
            "side": side,
            "entry": float(signal["entry_price"]),
            "tp_pips": float(signal["tp_pips"]),
            "sl_pips": float(signal["sl_pips"]),
            "units": units,
            "expires_index": index
            + max(1, math.ceil(int(signal.get("limit_expiry_seconds") or 60) / 60)),
        }
    else:
        entry = row["close"]
        arm.position = Position(
            side=side,
            entry=entry,
            tp=entry + float(signal["tp_pips"]) * PIP * (1 if side == "LONG" else -1),
            stop=entry - float(signal["sl_pips"]) * PIP * (1 if side == "LONG" else -1),
            units=units,
            opened_index=index,
            risk=float(signal["sl_pips"]) * PIP,
            remaining=units,
            favorable=entry,
        )
    arm.last_submission_epoch = row["epoch"]


def _metrics(values: list[float]) -> dict[str, Any]:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    equity = peak = max_dd = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "net_profit_jpy": round(sum(values), 2),
        "profit_factor": (
            round(gains / losses, 4) if losses else (None if gains else 0.0)
        ),
        "expectancy_jpy": round(sum(values) / len(values), 2) if values else None,
        "max_drawdown_jpy": round(max_dd, 2),
        "profit_giveback_ratio": (
            round(max(0.0, peak - equity) / peak, 4) if peak > 0 else None
        ),
        "trade_count": len(values),
    }


def replay(path: Path, policy: dict[str, Any], cost_pips: float) -> dict[str, Any]:
    candles = _load(path)
    arms = [Arm("BOT_ONLY", None), Arm("AI_INVENTORY", policy)]
    signal_engine = CausalM1Signal()
    closes: list[float] = []
    for index, row in enumerate(candles):
        closes.append(row["close"])
        raw_signal = signal_engine.add_completed_bar(
            _signal_bar(row), emit_signal=True
        )
        atr_pips = signal_engine.latest_atr_pips
        for arm in arms:
            _try_fill(arm, row, index)
            _manage(
                arm,
                row,
                index,
                closes,
                atr_pips,
                cost_pips,
            )
            if (
                raw_signal is not None
                and arm.position is None
                and arm.pending is None
            ):
                _submit(
                    arm,
                    raw_signal,
                    row,
                    index,
                    atr_pips,
                )
    return {
        arm.name: {
            **_metrics(arm.trades),
            "ai_policy_decision_count": arm.decisions,
            "ai_model_calls": 0,
            "ai_cost_jpy": 0.0,
            "end_open_position_excluded": arm.position is not None,
            "end_pending_order_excluded": arm.pending is not None,
        }
        for arm in arms
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candle", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-pips", type=float, default=1.1)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    policy_doc = json.loads(args.policy.read_text(encoding="utf-8"))
    if (
        policy_doc.get("authority") != AUTHORITY
        or policy_doc.get("contract")
        != "QR_DOJO_LEGACY_M1_AI_INVENTORY_POLICY_V1"
    ):
        raise SystemExit("invalid policy authority/contract")
    windows = {
        path.stem: replay(path, policy_doc["parameters"], args.cost_pips)
        for path in args.candle
    }
    body = {
        "contract": CONTRACT,
        "authority": AUTHORITY,
        "evidence_class": "LINEAGE_UNSEEN_DIAGNOSTIC_NOT_GLOBAL_HOLDOUT",
        "round_trip_cost_pips": args.cost_pips,
        "fixed_units_cap": 1000,
        "lot_increase_allowed": False,
        "future_data_allowed": False,
        "end_of_replay_liquidation_included": False,
        "windows": windows,
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
