#!/usr/bin/env python3
"""Fit a simple session/direction entry gate without looking at holdout results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluate", nargs="*", type=Path, default=[])
    parser.add_argument("--round-trip-cost-pips", type=float, default=1.1)
    parser.add_argument("--min-trades-per-train-window", type=int, default=10)
    parser.add_argument("--tune-inventory", action="store_true")
    parser.add_argument("--max-concurrent-grid", default="1,2,3,4")
    parser.add_argument("--cooldown-minutes-grid", default="0,2,3,5,8")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_trades(path: Path, strategy: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return [
        trade
        for trade in payload.get("trades", [])
        if trade.get("strategy") in {None, strategy}
    ]


def bucket(trade: dict[str, Any]) -> tuple[int, str]:
    return int(str(trade["entry_time"])[11:13]), str(trade["direction"])


def net(trade: dict[str, Any], cost_pips: float) -> float:
    units = abs(float(trade.get("units") or 10_000.0))
    if trade.get("pnl_jpy") is not None:
        gross_jpy = float(trade["pnl_jpy"])
    else:
        gross_jpy = float(trade.get("pnl_pips") or 0.0) * units * 0.01
    return gross_jpy - cost_pips * units * 0.01


def metrics(trades: list[dict[str, Any]], cost_pips: float) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda item: str(item["entry_time"]))
    values = [net(trade, cost_pips) for trade in ordered]
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = -sum(value for value in values if value < 0)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    giveback = None
    if peak > 0:
        giveback = max(0.0, (peak - equity) / peak)
    return {
        "trades": len(values),
        "net_pnl_jpy": round(sum(values), 2),
        "profit_factor": round(gross_profit / gross_loss, 4)
        if gross_loss
        else ("Infinity" if gross_profit else None),
        "expectancy_jpy": round(sum(values) / len(values), 2) if values else None,
        "max_drawdown_jpy": round(max_drawdown, 2),
        "profit_giveback_rate": round(giveback, 4)
        if giveback is not None
        else None,
    }


def inventory_filter(
    trades: list[dict[str, Any]], max_concurrent: int, cooldown_minutes: int
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    last_entry: datetime | None = None
    for trade in sorted(trades, key=lambda item: str(item["entry_time"])):
        entry = datetime.fromisoformat(str(trade["entry_time"]))
        active = sum(
            datetime.fromisoformat(str(previous["exit_time"])) > entry
            for previous in accepted
        )
        if active >= max_concurrent:
            continue
        if (
            last_entry is not None
            and (entry - last_entry).total_seconds() < cooldown_minutes * 60
        ):
            continue
        accepted.append(trade)
        last_entry = entry
    return accepted


def main() -> None:
    args = parse_args()
    train_by_window: list[dict[tuple[int, str], list[dict[str, Any]]]] = []
    for path in args.train:
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for trade in load_trades(path, args.strategy):
            grouped[bucket(trade)].append(trade)
        train_by_window.append(grouped)

    candidates = set.intersection(*(set(grouped) for grouped in train_by_window))
    selected = []
    for candidate in sorted(candidates):
        if all(
            len(grouped[candidate]) >= args.min_trades_per_train_window
            and sum(
                net(trade, args.round_trip_cost_pips)
                for trade in grouped[candidate]
            )
            > 0
            for grouped in train_by_window
        ):
            selected.append(candidate)

    def session_filter(path: Path) -> list[dict[str, Any]]:
        allowed = set(selected)
        return [
            trade
            for trade in load_trades(path, args.strategy)
            if bucket(trade) in allowed
        ]

    inventory_rule = {"max_concurrent": None, "cooldown_minutes": None}
    if args.tune_inventory and selected:
        grid = []
        for max_concurrent in map(int, args.max_concurrent_grid.split(",")):
            for cooldown_minutes in map(int, args.cooldown_minutes_grid.split(",")):
                window_metrics = [
                    metrics(
                        inventory_filter(
                            session_filter(path), max_concurrent, cooldown_minutes
                        ),
                        args.round_trip_cost_pips,
                    )
                    for path in args.train
                ]
                if all(
                    item["trades"] >= 5 and item["net_pnl_jpy"] > 0
                    for item in window_metrics
                ):
                    grid.append(
                        (
                            min(item["net_pnl_jpy"] for item in window_metrics),
                            sum(item["net_pnl_jpy"] for item in window_metrics),
                            -max_concurrent,
                            -cooldown_minutes,
                            max_concurrent,
                            cooldown_minutes,
                        )
                    )
        if grid:
            *_, max_concurrent, cooldown_minutes = max(grid)
            inventory_rule = {
                "max_concurrent": max_concurrent,
                "cooldown_minutes": cooldown_minutes,
            }

    def filter_path(path: Path) -> list[dict[str, Any]]:
        trades = session_filter(path)
        if inventory_rule["max_concurrent"] is None:
            return trades
        return inventory_filter(
            trades,
            int(inventory_rule["max_concurrent"]),
            int(inventory_rule["cooldown_minutes"]),
        )

    result = {
        "strategy_id": args.strategy,
        "fit_rule": {
            "method": "hour_direction_positive_in_every_train_window",
            "selected_utc_hour_direction": [
                {"hour_utc": hour, "direction": direction}
                for hour, direction in selected
            ],
            "min_trades_per_train_window": args.min_trades_per_train_window,
            "round_trip_cost_pips": args.round_trip_cost_pips,
            "holdout_used_for_fit": False,
            "inventory_rule": inventory_rule,
            "inventory_objective": (
                "maximize worst train-window net, then total train net"
                if args.tune_inventory
                else None
            ),
        },
        "train": {
            f"{path.parent.name}/{path.stem}": metrics(
                filter_path(path), args.round_trip_cost_pips
            )
            for path in args.train
        },
        "evaluation": {
            f"{path.parent.name}/{path.stem}": metrics(
                filter_path(path), args.round_trip_cost_pips
            )
            for path in args.evaluate
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
