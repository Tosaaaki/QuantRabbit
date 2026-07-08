#!/usr/bin/env python3
"""Build read-only EUR_USD SHORT BREAKOUT_FAILURE STOP HARVEST replay artifacts.

The replay uses observed OANDA transactions for the exact STOP samples and
independent local OANDA S5 bid/ask candles for trigger-to-TP path checks. It
never calls broker write endpoints and never grants scout or live permission.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / "data/eurusd_short_breakout_failure_stop_harvest_replay.json"
MD_OUT = ROOT / "docs/eurusd_short_breakout_failure_stop_harvest_replay.md"
LEDGER_DB = ROOT / "data/execution_ledger.db"
VEHICLE_SPLIT = ROOT / "data/eurusd_short_breakout_failure_vehicle_split_diagnosis.json"
MARKET_STOP_DIAGNOSIS = ROOT / "data/eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.json"
HISTORY_ROOT = ROOT / "logs/replay/oanda_history"

PAIR = "EUR_USD"
SIDE = "SHORT"
METHOD = "BREAKOUT_FAILURE"
ORDER_TYPE = "STOP"
TARGET_SHAPE = f"{PAIR}|{SIDE}|{METHOD}|{ORDER_TYPE}|HARVEST"
LANE_ID = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
PIP = 0.0001
S5_SECONDS = 5
S5_TOUCH_LAG_SECONDS = 60


@dataclass(frozen=True)
class Candle:
    time: datetime
    time_text: str
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float

    @property
    def spread_pips(self) -> float:
        return (self.ask_close - self.bid_close) / PIP


@dataclass(frozen=True)
class Touch:
    candle: Candle
    price: float
    order_id: str | None = None


@dataclass(frozen=True)
class TpScheduleItem:
    order_id: str
    active_from: datetime
    active_from_text: str
    price: float
    reason: str | None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    generated_at = _now()
    payload = build_payload(
        generated_at,
        root=args.root,
        history_root=args.history_root,
        ledger_db=args.ledger_db,
        vehicle_split_path=args.vehicle_split,
    )
    _write_json(args.output_json, payload)
    _write_text(args.output_md, _markdown(payload))
    print(f"wrote {_rel(args.output_json, args.root)}")
    print(f"wrote {_rel(args.output_md, args.root)}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--ledger-db", type=Path, default=LEDGER_DB)
    parser.add_argument("--vehicle-split", type=Path, default=VEHICLE_SPLIT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUT)
    return parser.parse_args(argv)


def build_payload(
    generated_at: str,
    *,
    root: Path = ROOT,
    history_root: Path = HISTORY_ROOT,
    ledger_db: Path = LEDGER_DB,
    vehicle_split_path: Path = VEHICLE_SPLIT,
) -> dict[str, Any]:
    trade_ids = _load_stop_trade_ids(vehicle_split_path)
    transactions = _load_oanda_transactions(ledger_db)
    samples = [_sample_from_transactions(trade_id, transactions) for trade_id in trade_ids]
    window_start = min(sample["entry_order_created_at_dt"] for sample in samples) - timedelta(minutes=5)
    window_end = max(sample["exit_timestamp_dt"] for sample in samples) + timedelta(minutes=5)
    candles, history_paths = _load_s5_bidask_candles(
        history_root,
        pair=PAIR,
        start=window_start,
        end=window_end,
    )
    if not candles:
        raise RuntimeError(f"no {PAIR} S5 BA candles found under {history_root}")

    replay_rows = [_replay_sample(sample, candles) for sample in samples]
    observed = _summary_from_rows(replay_rows, key="realized_pl_jpy")
    s5 = _s5_summary(replay_rows)
    mismatch = _s5_transaction_mismatch_summary(replay_rows)
    slippage = _slippage_sensitivity(replay_rows)
    invalidation = _invalidation_model(replay_rows)
    blockers = _remaining_blockers(s5)
    status = (
        "STOP_HARVEST_EXACT_S5_BIDASK_REPLAY_PASSED_STILL_BLOCKED"
        if s5["s5_trigger_and_tp_path_passed"]
        else "STOP_HARVEST_EXACT_S5_BIDASK_REPLAY_FAILED_BLOCKED"
    )
    bidask_status = (
        "PASSED_WITH_S5_TRIGGER_AND_TP_PATH_7_OF_7_STOP_SAMPLES_STILL_SCOUT_BLOCKED"
        if s5["s5_trigger_and_tp_path_passed"] and s5["sample_count"] == 7
        else "S5_TRIGGER_OR_TP_PATH_INCOMPLETE_STILL_BLOCKED"
    )
    return {
        "status": status,
        "target_shape": TARGET_SHAPE,
        "generated_at_utc": generated_at,
        "read_only": True,
        "live_side_effects": [],
        "live_permission_allowed": False,
        "scout_candidate_after_replay": False,
        "trigger_price_available": all(row["trigger_price"] is not None for row in replay_rows),
        "bidask_replay_status": bidask_status,
        "observed_stop_summary": observed,
        "s5_path_replay_summary": s5,
        "s5_transaction_mismatch_summary": mismatch,
        "slippage_sensitivity": slippage,
        "invalidation_model": invalidation,
        "max_loss_cap_model": _max_loss_cap_model(replay_rows),
        "net_expectancy_after_bidask_slippage": observed["expectancy_jpy_per_trade"],
        "replay_sample_count": len(replay_rows),
        "replay_wins": observed["wins"],
        "replay_losses": observed["losses"],
        "stop_vehicle_status": (
            "EXACT_S5_PATH_REPLAY_POSITIVE_BUT_SCOUT_AND_LIVE_BLOCKED"
            if s5["s5_trigger_and_tp_path_passed"]
            else "OBSERVED_TRANSACTION_POSITIVE_BUT_EXACT_S5_PATH_REPLAY_FAILED"
        ),
        "remaining_blockers": blockers,
        "next_read_only_actions": [
            "Define STOP-specific pre-trigger stop-chase and post-trigger invalidation thresholds from a larger out-of-sample S5 set.",
            "Define max trigger slippage and max loss cap from current risk budget before SCOUT_READY_CHECK.",
            "Keep STOP_HARVEST separate from LIMIT_HARVEST and MARKET_HARVEST proof vehicles.",
            "Refresh proof queue, 4x planner, harvest path, active contract, and operator-review material read-only after STOP contract fields exist.",
            "Keep NEGATIVE_EXPECTANCY_ACTIVE, MARKET_CLOSE_LEAK_PRESENT, and MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE visible until refreshed artifacts clear them.",
        ],
        "do_not_do": _do_not_do(),
        "source_artifacts": [
            _rel(ledger_db, root),
            _rel(vehicle_split_path, root),
            _rel(MARKET_STOP_DIAGNOSIS, root),
            *[_rel(path, root) for path in history_paths],
        ],
        "proof_boundary_checks": {
            "stop_samples_only": True,
            "market_samples_mixed_in": False,
            "limit_samples_mixed_in": False,
            "market_close_losses_mixed_in": False,
            "sample_trade_ids": [row["trade_id"] for row in replay_rows],
        },
        "replay_method": {
            "kind": "independent_s5_bidask_trigger_to_tp_path_replay",
            "short_stop_trigger_stream": "bid_low_lte_stop_trigger",
            "short_take_profit_stream": "ask_low_lte_active_tp",
            "tp_schedule_model": "OANDA TAKE_PROFIT_ORDER replacements by trade_id; active TP changes over time.",
            "observed_transaction_cross_check": "OANDA ORDER_FILL fullPrice bid/ask and realized P/L remain the cost-inclusive ledger truth.",
            "result_interpretation": _result_interpretation(observed, s5),
        },
        "replay_rows": replay_rows,
    }


def _load_stop_trade_ids(path: Path) -> list[str]:
    payload = _load_json(path)
    split = payload.get("vehicle_split") if isinstance(payload.get("vehicle_split"), dict) else {}
    rows = split.get("stop_order_samples") if isinstance(split.get("stop_order_samples"), list) else []
    out = [str(row.get("trade_id")) for row in rows if isinstance(row, dict) and row.get("trade_id")]
    if not out:
        raise RuntimeError(f"no stop_order_samples found in {path}")
    return out


def _load_oanda_transactions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "select transaction_id, raw_json from oanda_transactions order by cast(transaction_id as integer)"
        ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for transaction_id, raw in rows:
        payload = json.loads(raw)
        out[str(transaction_id)] = payload
    return out


def _sample_from_transactions(trade_id: str, transactions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entry = transactions.get(trade_id)
    if not entry:
        raise RuntimeError(f"missing entry fill transaction for trade {trade_id}")
    if entry.get("type") != "ORDER_FILL" or entry.get("reason") != "STOP_ORDER":
        raise RuntimeError(f"trade {trade_id} is not a STOP_ORDER entry fill")
    entry_order_id = str(entry.get("orderID") or "")
    stop_order = transactions.get(entry_order_id)
    if not stop_order or stop_order.get("type") != "STOP_ORDER":
        raise RuntimeError(f"missing STOP_ORDER creation {entry_order_id} for trade {trade_id}")
    exit_fill = _exit_fill_for_trade(trade_id, transactions.values())
    final_tp_order_id = str(exit_fill.get("orderID") or "")
    final_tp = transactions.get(final_tp_order_id)
    tp_schedule = _tp_schedule_for_trade(trade_id, transactions.values())
    return {
        "trade_id": trade_id,
        "lane_id": LANE_ID,
        "entry_order_id": entry_order_id,
        "entry_order_created_at_utc": str(stop_order.get("time")),
        "entry_order_created_at_dt": _parse_time(str(stop_order.get("time"))),
        "trigger_mode": "TOP_OF_BOOK",
        "trigger_condition": stop_order.get("triggerCondition") or "DEFAULT",
        "trigger_price": _float(stop_order.get("price")),
        "entry_fill_transaction_id": str(entry.get("id") or trade_id),
        "entry_timestamp_utc": str(entry.get("time")),
        "entry_timestamp_dt": _parse_time(str(entry.get("time"))),
        "entry_price": _float(entry.get("price")),
        "entry_full_price": _full_price(entry),
        "entry_half_spread_cost_jpy": _float(entry.get("halfSpreadCost")),
        "units": int(float(entry.get("units") or 0)),
        "initial_take_profit_on_fill_price": tp_schedule[0].price if tp_schedule else None,
        "tp_schedule": tp_schedule,
        "final_tp_order_id": final_tp_order_id,
        "final_tp_order_price": _float((final_tp or {}).get("price") or exit_fill.get("price")),
        "exit_transaction_id": str(exit_fill.get("id") or ""),
        "exit_timestamp_utc": str(exit_fill.get("time")),
        "exit_timestamp_dt": _parse_time(str(exit_fill.get("time"))),
        "exit_price": _float(exit_fill.get("price")),
        "exit_full_price": _full_price(exit_fill),
        "exit_half_spread_cost_jpy": _float(exit_fill.get("halfSpreadCost")),
        "exit_reason": str(exit_fill.get("reason") or ""),
        "realized_pl_jpy": _float(exit_fill.get("pl")),
        "quote_pl_usd": _float(exit_fill.get("quotePL")),
    }


def _exit_fill_for_trade(trade_id: str, transactions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for tx in transactions:
        if tx.get("type") != "ORDER_FILL" or tx.get("reason") != "TAKE_PROFIT_ORDER":
            continue
        for close in tx.get("tradesClosed") or []:
            if str(close.get("tradeID")) == str(trade_id):
                matches.append(tx)
                break
    if not matches:
        raise RuntimeError(f"missing TAKE_PROFIT_ORDER exit fill for trade {trade_id}")
    return sorted(matches, key=lambda item: _parse_time(str(item.get("time"))))[0]


def _tp_schedule_for_trade(trade_id: str, transactions: Iterable[dict[str, Any]]) -> list[TpScheduleItem]:
    rows = []
    for tx in transactions:
        if tx.get("type") == "TAKE_PROFIT_ORDER" and str(tx.get("tradeID")) == str(trade_id):
            price = _float(tx.get("price"))
            if price is None:
                continue
            rows.append(
                TpScheduleItem(
                    order_id=str(tx.get("id") or ""),
                    active_from=_parse_time(str(tx.get("time"))),
                    active_from_text=str(tx.get("time")),
                    price=price,
                    reason=str(tx.get("reason") or ""),
                )
            )
    rows.sort(key=lambda item: item.active_from)
    if not rows:
        raise RuntimeError(f"missing TAKE_PROFIT_ORDER schedule for trade {trade_id}")
    return rows


def _load_s5_bidask_candles(
    history_root: Path,
    *,
    pair: str,
    start: datetime,
    end: datetime,
) -> tuple[list[Candle], list[Path]]:
    files = sorted(history_root.rglob(f"{pair}_S5_BA_*.jsonl*"))
    candles_by_time: dict[datetime, Candle] = {}
    used_paths: list[Path] = []
    for path in files:
        path_used = False
        with _open_jsonl(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("pair") != pair or row.get("granularity") != "S5" or row.get("price") != "BA":
                    continue
                time = _parse_time(str(row.get("time")))
                if time < start or time > end:
                    continue
                candle = _candle_from_row(row)
                candles_by_time[candle.time] = candle
                path_used = True
        if path_used:
            used_paths.append(path)
    return [candles_by_time[key] for key in sorted(candles_by_time)], used_paths


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _candle_from_row(row: dict[str, Any]) -> Candle:
    bid = row.get("bid") if isinstance(row.get("bid"), dict) else {}
    ask = row.get("ask") if isinstance(row.get("ask"), dict) else {}
    return Candle(
        time=_parse_time(str(row.get("time"))),
        time_text=str(row.get("time")),
        bid_open=float(bid["o"]),
        bid_high=float(bid["h"]),
        bid_low=float(bid["l"]),
        bid_close=float(bid["c"]),
        ask_open=float(ask["o"]),
        ask_high=float(ask["h"]),
        ask_low=float(ask["l"]),
        ask_close=float(ask["c"]),
    )


def _replay_sample(sample: dict[str, Any], candles: Sequence[Candle]) -> dict[str, Any]:
    trigger = _first_short_stop_trigger(
        candles,
        start=sample["entry_order_created_at_dt"],
        end=sample["exit_timestamp_dt"],
        trigger_price=sample["trigger_price"],
    )
    tp = _first_short_tp_touch(
        candles,
        start=sample["entry_timestamp_dt"],
        end=sample["exit_timestamp_dt"] + timedelta(seconds=S5_TOUCH_LAG_SECONDS),
        schedule=sample["tp_schedule"],
    )
    entry_candle = _candle_at_or_before(candles, sample["entry_timestamp_dt"])
    exit_candle = _candle_at_or_before(candles, sample["exit_timestamp_dt"])
    path_start = trigger.candle.time if trigger else _ceil_s5(sample["entry_timestamp_dt"])
    path_end = tp.candle.time if tp else _ceil_s5(sample["exit_timestamp_dt"])
    path_candles = [
        candle for candle in candles if candle.time >= path_start and candle.time <= path_end
    ]
    pre_trigger_candles = (
        [candle for candle in candles if candle.time >= _ceil_s5(sample["entry_order_created_at_dt"]) and candle.time <= trigger.candle.time]
        if trigger
        else []
    )
    realized = sample["realized_pl_jpy"] or 0.0
    pips_to_exit = _short_pips(sample["entry_price"], sample["exit_price"])
    pip_value = realized / pips_to_exit if pips_to_exit else None
    trigger_price = sample["trigger_price"]
    holding_max_ask_high = max((candle.ask_high for candle in path_candles), default=None)
    holding_min_ask_low = min((candle.ask_low for candle in path_candles), default=None)
    pre_max_ask_high = max((candle.ask_high for candle in pre_trigger_candles), default=None)
    return {
        "trade_id": sample["trade_id"],
        "lane_id": sample["lane_id"],
        "source": "data/execution_ledger.db:oanda_transactions + logs/replay/oanda_history S5 BA",
        "entry_order_id": sample["entry_order_id"],
        "entry_order_type": "STOP_ORDER",
        "entry_order_created_at_utc": sample["entry_order_created_at_utc"],
        "trigger_mode": sample["trigger_mode"],
        "trigger_condition": sample["trigger_condition"],
        "trigger_price": _price(trigger_price),
        "entry_fill_transaction_id": sample["entry_fill_transaction_id"],
        "entry_timestamp_utc": sample["entry_timestamp_utc"],
        "entry_price": _price(sample["entry_price"]),
        "entry_bid": _price((sample["entry_full_price"] or {}).get("bid")),
        "entry_ask": _price((sample["entry_full_price"] or {}).get("ask")),
        "entry_spread_pips": _round(_spread_pips(sample["entry_full_price"])),
        "entry_half_spread_cost_jpy": _round(sample["entry_half_spread_cost_jpy"], 4),
        "observed_trigger_fill_slippage_pips": _round(
            max(0.0, ((trigger_price or 0.0) - (sample["entry_price"] or 0.0)) / PIP)
        ),
        "s5_first_trigger_touch_utc": trigger.candle.time_text if trigger else None,
        "s5_trigger_touch_bid_low": _price(trigger.candle.bid_low if trigger else None),
        "s5_trigger_touch_lag_seconds_from_order": (
            _round((trigger.candle.time - sample["entry_order_created_at_dt"]).total_seconds(), 3)
            if trigger
            else None
        ),
        "s5_fill_lag_seconds_from_trigger_touch": (
            _round((sample["entry_timestamp_dt"] - trigger.candle.time).total_seconds(), 3)
            if trigger
            else None
        ),
        "s5_trigger_candle_bid_low_below_trigger_pips": _round(
            max(0.0, ((trigger_price or 0.0) - trigger.candle.bid_low) / PIP) if trigger else None
        ),
        "entry_candle_start_utc": entry_candle.time_text if entry_candle else None,
        "entry_candle_bid_low": _price(entry_candle.bid_low if entry_candle else None),
        "entry_candle_trigger_touch": (
            bool(entry_candle and trigger_price is not None and entry_candle.bid_low <= trigger_price)
        ),
        "units": sample["units"],
        "units_abs": abs(sample["units"]),
        "initial_take_profit_on_fill_price": _price(sample["initial_take_profit_on_fill_price"]),
        "final_tp_order_id": sample["final_tp_order_id"],
        "final_tp_order_price": _price(sample["final_tp_order_price"]),
        "tp_schedule": [
            {
                "order_id": item.order_id,
                "active_from_utc": item.active_from_text,
                "price": _price(item.price),
                "reason": item.reason,
            }
            for item in sample["tp_schedule"]
        ],
        "s5_first_tp_touch_after_trigger_utc": tp.candle.time_text if tp else None,
        "s5_first_tp_touch_order_id": tp.order_id if tp else None,
        "s5_first_tp_touch_price": _price(tp.price if tp else None),
        "s5_first_tp_touch_ask_low": _price(tp.candle.ask_low if tp else None),
        "s5_tp_touch_lag_seconds_from_exit": (
            _round((tp.candle.time - sample["exit_timestamp_dt"]).total_seconds(), 3)
            if tp
            else None
        ),
        "exit_transaction_id": sample["exit_transaction_id"],
        "exit_timestamp_utc": sample["exit_timestamp_utc"],
        "exit_price": _price(sample["exit_price"]),
        "exit_bid": _price((sample["exit_full_price"] or {}).get("bid")),
        "exit_ask": _price((sample["exit_full_price"] or {}).get("ask")),
        "exit_spread_pips": _round(_spread_pips(sample["exit_full_price"])),
        "exit_half_spread_cost_jpy": _round(sample["exit_half_spread_cost_jpy"], 4),
        "exit_candle_start_utc": exit_candle.time_text if exit_candle else None,
        "exit_candle_ask_low": _price(exit_candle.ask_low if exit_candle else None),
        "exit_candle_tp_touch": (
            bool(exit_candle and sample["final_tp_order_price"] is not None and exit_candle.ask_low <= sample["final_tp_order_price"])
        ),
        "exit_reason": sample["exit_reason"],
        "realized_pl_jpy": _round(sample["realized_pl_jpy"], 4),
        "quote_pl_usd": _round(sample["quote_pl_usd"], 6),
        "pips_to_exit": _round(pips_to_exit),
        "pip_value_jpy_per_pip": _round(pip_value, 6),
        "holding_min_ask_low": _price(holding_min_ask_low),
        "holding_max_ask_high": _price(holding_max_ask_high),
        "post_trigger_mfe_pips": _round(_positive_pips((trigger_price or 0.0) - (holding_min_ask_low or trigger_price or 0.0))),
        "post_trigger_mae_pips": _round(_positive_pips((holding_max_ask_high or trigger_price or 0.0) - (trigger_price or 0.0))),
        "observed_path_adverse_jpy_at_mae": _round(
            (_positive_pips((holding_max_ask_high or trigger_price or 0.0) - (trigger_price or 0.0)) * pip_value)
            if pip_value is not None
            else None,
            4,
        ),
        "pre_trigger_max_ask_high_above_trigger_pips": _round(
            _positive_pips((pre_max_ask_high or trigger_price or 0.0) - (trigger_price or 0.0))
        ),
        "s5_spread_pips_min": _round(min((candle.spread_pips for candle in path_candles), default=math.nan)),
        "s5_spread_pips_max": _round(max((candle.spread_pips for candle in path_candles), default=math.nan)),
        "s5_spread_pips_avg": _round(
            (sum(candle.spread_pips for candle in path_candles) / len(path_candles))
            if path_candles
            else None
        ),
        "s5_trigger_touch_pass": trigger is not None,
        "s5_tp_touch_after_trigger_pass": tp is not None,
        "s5_replay_win": bool(trigger and tp and realized > 0.0),
        "s5_replay_loss": not bool(trigger and tp and realized > 0.0),
        "observed_bidask_cost_inclusive_pass": realized > 0.0,
        "s5_tp_miss_distance_pips": _round(
            _positive_pips((holding_min_ask_low or sample["final_tp_order_price"] or 0.0) - (sample["final_tp_order_price"] or 0.0))
            if tp is None and holding_min_ask_low is not None and sample["final_tp_order_price"] is not None
            else None
        ),
        "market_close_mixed_in": False,
        "market_sample_mixed_in": False,
        "limit_sample_mixed_in": False,
    }


def _first_short_stop_trigger(
    candles: Sequence[Candle],
    *,
    start: datetime,
    end: datetime,
    trigger_price: float | None,
) -> Touch | None:
    if trigger_price is None:
        return None
    first_time = _ceil_s5(start)
    for candle in candles:
        if candle.time < first_time:
            continue
        if candle.time > end:
            break
        if candle.bid_low <= trigger_price:
            return Touch(candle=candle, price=trigger_price)
    return None


def _first_short_tp_touch(
    candles: Sequence[Candle],
    *,
    start: datetime,
    end: datetime,
    schedule: Sequence[TpScheduleItem],
) -> Touch | None:
    first_time = _ceil_s5(start)
    for candle in candles:
        if candle.time < first_time:
            continue
        if candle.time > end:
            break
        active = _active_tp(schedule, candle.time)
        if active and candle.ask_low <= active.price:
            return Touch(candle=candle, price=active.price, order_id=active.order_id)
    return None


def _active_tp(schedule: Sequence[TpScheduleItem], at: datetime) -> TpScheduleItem | None:
    active = None
    for item in schedule:
        if item.active_from <= at:
            active = item
        else:
            break
    return active


def _candle_at_or_before(candles: Sequence[Candle], at: datetime) -> Candle | None:
    target = _floor_s5(at)
    for candle in candles:
        if candle.time == target:
            return candle
    return None


def _summary_from_rows(rows: Sequence[dict[str, Any]], *, key: str) -> dict[str, Any]:
    values = [float(row.get(key) or 0.0) for row in rows]
    wins = len([value for value in values if value > 0.0])
    losses = len([value for value in values if value <= 0.0])
    net = sum(values)
    return {
        "wins": wins,
        "losses": losses,
        "net_jpy": _round(net, 4),
        "sample_count": len(values),
        "expectancy_jpy_per_trade": _round(net / len(values), 4) if values else None,
    }


def _s5_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    trigger_pass = len([row for row in rows if row.get("s5_trigger_touch_pass")])
    tp_pass = len([row for row in rows if row.get("s5_tp_touch_after_trigger_pass")])
    wins = len([row for row in rows if row.get("s5_replay_win")])
    losses = len(rows) - wins
    mae_values = [float(row.get("post_trigger_mae_pips") or 0.0) for row in rows]
    mfe_values = [float(row.get("post_trigger_mfe_pips") or 0.0) for row in rows]
    return {
        "sample_count": len(rows),
        "s5_trigger_touch_count": trigger_pass,
        "s5_tp_touch_after_trigger_count": tp_pass,
        "s5_trigger_missing_trade_ids": [row["trade_id"] for row in rows if not row.get("s5_trigger_touch_pass")],
        "s5_tp_path_missing_trade_ids": [row["trade_id"] for row in rows if not row.get("s5_tp_touch_after_trigger_pass")],
        "s5_wins": wins,
        "s5_losses": losses,
        "s5_trigger_and_tp_path_passed": bool(rows) and trigger_pass == len(rows) and tp_pass == len(rows) and losses == 0,
        "post_trigger_mae_pips": _dist(mae_values),
        "post_trigger_mfe_pips": _dist(mfe_values),
        "max_observed_path_adverse_jpy": _round(
            max((float(row.get("observed_path_adverse_jpy_at_mae") or 0.0) for row in rows), default=0.0),
            4,
        ),
        "strict_same_candle_trigger_touch_count": len([row for row in rows if row.get("entry_candle_trigger_touch")]),
        "strict_same_candle_tp_touch_count": len([row for row in rows if row.get("exit_candle_tp_touch")]),
        "timestamp_alignment_status": "S5_PATH_TOUCH_ORDER_CONFIRMED_WITH_FILL_LAG_VISIBLE",
        "allowed_s5_touch_lag_seconds_after_exit": S5_TOUCH_LAG_SECONDS,
    }


def _s5_transaction_mismatch_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    missed_tp = [row for row in rows if not row.get("s5_tp_touch_after_trigger_pass")]
    distances = [
        float(row["s5_tp_miss_distance_pips"])
        for row in missed_tp
        if row.get("s5_tp_miss_distance_pips") is not None
    ]
    return {
        "status": (
            "NO_S5_TRANSACTION_PATH_MISMATCH_DETECTED"
            if not missed_tp
            else "OANDA_TRANSACTION_TP_FILLS_NOT_FULLY_RECONSTRUCTED_BY_S5_BA_CANDLES"
        ),
        "observed_transaction_wins": len([row for row in rows if row.get("observed_bidask_cost_inclusive_pass")]),
        "s5_tp_missing_count": len(missed_tp),
        "s5_tp_missing_trade_ids": [row["trade_id"] for row in missed_tp],
        "s5_tp_miss_distance_pips": _dist(distances),
        "interpretation": (
            "Observed OANDA ORDER_FILL fullPrice confirms the broker-side TP exits, but independent "
            "S5 bid/ask candles do not reconstruct every active-TP touch within the allowed lag. "
            "This is a hard SCOUT blocker, not evidence to relax the gate."
            if missed_tp
            else "Observed OANDA ORDER_FILL fullPrice and independent S5 bid/ask path agree for all STOP samples."
        ),
    }


def _result_interpretation(observed: dict[str, Any], s5: dict[str, Any]) -> str:
    if s5["s5_trigger_and_tp_path_passed"]:
        return (
            "Observed transaction replay and independent S5 path replay are both positive, but 7 samples "
            "and missing scout/live invalidation contracts keep permission blocked."
        )
    return (
        f"Observed OANDA transaction replay is positive ({observed['wins']}/{observed['sample_count']} wins), "
        f"but independent S5 path replay only reconstructs {s5['s5_tp_touch_after_trigger_count']}/"
        f"{s5['sample_count']} active TP touches after confirming {s5['s5_trigger_touch_count']}/"
        f"{s5['sample_count']} STOP triggers. Treat STOP_HARVEST as failed/blocked before SCOUT."
    )


def _slippage_sensitivity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scenarios = []
    per_trade_break_even: dict[str, float] = {}
    for row in rows:
        pip_value = float(row.get("pip_value_jpy_per_pip") or 0.0)
        realized = float(row.get("realized_pl_jpy") or 0.0)
        if pip_value > 0:
            per_trade_break_even[str(row["trade_id"])] = _round(realized / pip_value)
    for extra in [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
        values = []
        for row in rows:
            pip_value = float(row.get("pip_value_jpy_per_pip") or 0.0)
            values.append(float(row.get("realized_pl_jpy") or 0.0) - (extra * pip_value))
        net = sum(values)
        scenarios.append(
            {
                "total_extra_adverse_slippage_pips": extra,
                "net_jpy": _round(net, 4),
                "expectancy_jpy_per_trade": _round(net / len(values), 4) if values else None,
                "wins": len([value for value in values if value > 0.0]),
                "losses": len([value for value in values if value <= 0.0]),
                "min_trade_jpy": _round(min(values), 4) if values else None,
            }
        )
    return {
        "basis": (
            "Observed OANDA transaction bid/ask plus independent S5 trigger/TP path; "
            "sensitivity subtracts extra total adverse pips from each trade using realized JPY per pip."
        ),
        "observed_trigger_fill_slippage_pips": _dist(
            [float(row.get("observed_trigger_fill_slippage_pips") or 0.0) for row in rows]
        ),
        "s5_trigger_candle_low_beyond_trigger_pips": _dist(
            [float(row.get("s5_trigger_candle_bid_low_below_trigger_pips") or 0.0) for row in rows]
        ),
        "entry_spread_pips": _dist([float(row.get("entry_spread_pips") or 0.0) for row in rows]),
        "exit_spread_pips": _dist([float(row.get("exit_spread_pips") or 0.0) for row in rows]),
        "break_even_extra_total_adverse_slippage_pips": {
            **_dist(list(per_trade_break_even.values())),
            "per_trade": per_trade_break_even,
        },
        "scenarios": scenarios,
    }


def _invalidation_model(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "S5_PATH_REPLAYED_BUT_SCOUT_INVALIDATION_NOT_DEFINED",
        "can_define_observed_replay_invalidations": True,
        "can_define_scout_or_live_invalidations": False,
        "observed_replay_rules": [
            "Accept only EUR_USD SHORT BREAKOUT_FAILURE rows with entry_order_type=STOP_ORDER and exit_reason=TAKE_PROFIT_ORDER.",
            "Require independent S5 bid/ask trigger touch: SHORT STOP trigger passes only when bid_low <= trigger after order creation.",
            "Require independent S5 bid/ask TP touch: SHORT attached TP passes only when ask_low <= active TAKE_PROFIT_ORDER after trigger.",
            "Replay the active TP schedule from OANDA TAKE_PROFIT_ORDER creation/replacement rows; do not assume final TP was active from entry.",
            "Reject MARKET_ORDER/LIMIT_ORDER rows and any MARKET_ORDER_TRADE_CLOSE exit from this STOP proof.",
            "Mark replay invalid if trigger touch, active TP touch, OANDA bid/ask, or positive realized P/L is missing.",
        ],
        "observed_path_metrics": {
            "post_trigger_mae_pips": _dist([float(row.get("post_trigger_mae_pips") or 0.0) for row in rows]),
            "pre_trigger_max_ask_high_above_trigger_pips": _dist(
                [float(row.get("pre_trigger_max_ask_high_above_trigger_pips") or 0.0) for row in rows]
            ),
        },
        "scout_gap": (
            "The 7 historical OANDA transaction winners are useful broker-side evidence, but they do not by "
            "themselves prove a complete independent S5 trigger-to-TP path or define a production pre-trigger "
            "stop-chase cutoff, max trigger slippage, post-trigger invalidation level, or max loss cap."
        ),
        "required_next_contract_fields": [
            "pre_trigger_stop_chase_filter_from_out_of_sample_S5_path",
            "post_trigger_invalidation_before_TP_from_out_of_sample_S5_path",
            "max_allowed_trigger_slippage_pips_from_replay_distribution",
            "equity_derived_max_loss_cap_jpy_for_scout",
            "forecast_support_and_stop_entry_timing_rule_for_STOP_vehicle",
        ],
    }


def _max_loss_cap_model(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_trade = {
        str(row["trade_id"]): {
            "observed_post_trigger_mae_pips": row.get("post_trigger_mae_pips"),
            "observed_path_adverse_jpy_at_mae": row.get("observed_path_adverse_jpy_at_mae"),
        }
        for row in rows
    }
    values = [
        float(row.get("observed_path_adverse_jpy_at_mae") or 0.0)
        for row in rows
    ]
    return {
        "status": "OBSERVED_PATH_ONLY_NOT_SCOUT_READY",
        "observed_worst_path_adverse_jpy": _round(max(values), 4) if values else None,
        "per_trade": per_trade,
        "production_max_loss_cap_defined": False,
        "reason": (
            "Observed MAE is useful for scout design, but production cap must be equity-derived "
            "and validated through RiskEngine/LiveOrderGateway before any SCOUT_READY_CHECK."
        ),
    }


def _remaining_blockers(s5_summary: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not s5_summary.get("s5_trigger_and_tp_path_passed"):
        blockers.append(
            {
                "code": "STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED",
                "status": "BLOCKING_SCOUT_AND_LIVE_PERMISSION",
                "evidence": "At least one STOP sample lacks independent S5 trigger touch or active TP touch.",
            }
        )
    if s5_summary.get("s5_tp_path_missing_trade_ids"):
        blockers.append(
            {
                "code": "S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS",
                "status": "BLOCKING_SCOUT_AND_LIVE_PERMISSION",
                "evidence": (
                    "OANDA transaction fullPrice confirms TP exits, but independent S5 bid/ask candles "
                    f"miss active TP touches for trades {s5_summary['s5_tp_path_missing_trade_ids']}."
                ),
            }
        )
    blockers.extend(
        [
            {
                "code": "STOP_SAMPLE_COUNT_THIN_FOR_LIVE_GRADE",
                "status": "BLOCKING_LIVE_PERMISSION",
                "evidence": "STOP_HARVEST has only 7 STOP samples; even a fully reconstructed replay would not be live-grade proof.",
            },
            {
                "code": "STOP_TRIGGER_INVALIDATION_NOT_SCOUT_READY",
                "status": "BLOCKING_SCOUT",
                "evidence": "Pre-trigger stop-chase, max trigger slippage, post-trigger invalidation, and max loss cap are not production-defined.",
            },
            {
                "code": "GLOBAL_NEGATIVE_EXPECTANCY_ACTIVE",
                "status": "VISIBLE_GLOBAL_BLOCKER",
                "evidence": "This STOP packet does not clear global capture-economics or self-improvement blockers.",
            },
            {
                "code": "MARKET_CLOSE_LEAK_PRESENT_EXCLUDED",
                "status": "VISIBLE_TARGET_SHAPE_BLOCKER",
                "evidence": "Market-close leakage remains excluded from this STOP/TP proof and must not be mixed into HARVEST proof.",
            },
            {
                "code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                "status": "VISIBLE_GLOBAL_BLOCKER",
                "evidence": "Existing profitability/harvest artifacts keep the month-scale residual blocker visible.",
            },
            {
                "code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                "status": "BLOCKING_ROUTING",
                "evidence": "Normal routing remains blocked until guardian receipt/operator-review state is cleared by its own contract.",
            },
            {
                "code": "NO_LIVE_READY_PORTFOLIO",
                "status": "BLOCKING_LIVE_PERMISSION",
                "evidence": "Portfolio path still has no live-ready permission for STOP_HARVEST.",
            },
            {
                "code": "NO_FRESH_GATEWAY_PERMISSION",
                "status": "BLOCKING_LIVE_PERMISSION",
                "evidence": "No verifier/gateway packet exists for STOP_HARVEST.",
            },
        ]
    )
    return blockers


def _do_not_do() -> list[str]:
    return [
        "Do not send live orders.",
        "Do not stage live orders.",
        "Do not cancel orders.",
        "Do not close positions.",
        "Do not modify broker state, TP, SL, or launchd.",
        "Do not relax gates.",
        "Do not mix STOP samples into LIMIT proof.",
        "Do not mix MARKET samples into STOP proof.",
        "Do not mix market-close losses into TP/HARVEST proof.",
        "Do not hide negative expectancy or month-scale replay negatives.",
        "Do not backsolve lot size from the 4x deficit.",
        "Do not infer or invent an operator decision.",
        "Do not print, copy, or persist secrets.",
    ]


def _markdown(payload: dict[str, Any]) -> str:
    s5 = payload["s5_path_replay_summary"]
    observed = payload["observed_stop_summary"]
    mismatch = payload["s5_transaction_mismatch_summary"]
    verdict_text = (
        f"The STOP-only observed OANDA transaction replay is positive: `{observed['wins']}` wins, "
        f"`{observed['losses']}` losses, net `{observed['net_jpy']}` JPY, expectancy "
        f"`{observed['expectancy_jpy_per_trade']}` JPY/trade. Independent S5 bid/ask path replay "
        f"confirms `{s5['s5_trigger_touch_count']}/{s5['sample_count']}` STOP triggers but only "
        f"`{s5['s5_tp_touch_after_trigger_count']}/{s5['sample_count']}` active TP touches. "
        "This is not SCOUT-ready and does not create live permission."
        if not s5["s5_trigger_and_tp_path_passed"]
        else (
            f"The STOP-only observed OANDA transaction replay and independent S5 bid/ask path replay "
            f"are both positive: `{observed['wins']}` wins, `{observed['losses']}` losses, net "
            f"`{observed['net_jpy']}` JPY, expectancy `{observed['expectancy_jpy_per_trade']}` "
            "JPY/trade. This does not create scout or live permission because STOP-specific "
            "invalidation, slippage, max-loss, guardian/operator, and portfolio gates remain blocked."
        )
    )
    lines = [
        "# EUR_USD SHORT BREAKOUT_FAILURE STOP HARVEST Replay",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{payload['status']}`",
        f"- Target shape: `{payload['target_shape']}`",
        f"- Read-only: `{payload['read_only']}`",
        f"- Live permission allowed: `{payload['live_permission_allowed']}`",
        f"- Scout candidate after replay: `{payload['scout_candidate_after_replay']}`",
        f"- S5 trigger/TP path passed: `{s5['s5_trigger_and_tp_path_passed']}`",
        "",
        verdict_text,
        "",
        "## Boundary",
        "",
        f"- STOP samples only: `{payload['proof_boundary_checks']['stop_samples_only']}`",
        f"- MARKET samples mixed into STOP proof: `{payload['proof_boundary_checks']['market_samples_mixed_in']}`",
        f"- LIMIT samples mixed into STOP proof: `{payload['proof_boundary_checks']['limit_samples_mixed_in']}`",
        f"- Market-close losses mixed into HARVEST proof: `{payload['proof_boundary_checks']['market_close_losses_mixed_in']}`",
        "",
        "## Replay Basis",
        "",
        "SHORT STOP trigger is checked on S5 `bid_low <= trigger`. SHORT attached TP is checked on S5 "
        "`ask_low <= active TAKE_PROFIT_ORDER`. TP replacements are replayed as a schedule; the final TP is "
        "not assumed to have been active from entry.",
        "",
        "## S5 Summary",
        "",
        f"- Trigger touches: `{s5['s5_trigger_touch_count']}/{s5['sample_count']}`",
        f"- TP touches after trigger: `{s5['s5_tp_touch_after_trigger_count']}/{s5['sample_count']}`",
        f"- S5 wins/losses: `{s5['s5_wins']}/{s5['s5_losses']}`",
        f"- TP path missing trade IDs: `{s5['s5_tp_path_missing_trade_ids']}`",
        f"- Transaction/S5 mismatch status: `{mismatch['status']}`",
        f"- S5 TP miss distance pips: `{mismatch['s5_tp_miss_distance_pips']}`",
        f"- Post-trigger MAE pips: `{s5['post_trigger_mae_pips']}`",
        f"- Post-trigger MFE pips: `{s5['post_trigger_mfe_pips']}`",
        f"- Max observed path adverse JPY: `{s5['max_observed_path_adverse_jpy']}`",
        "",
        "## Samples",
        "",
        "| trade_id | stop_order | trigger | S5 trigger | S5 TP | realized JPY | MAE pips | MFE pips |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in payload["replay_rows"]:
        lines.append(
            f"| {row['trade_id']} | {row['entry_order_id']} | {row['trigger_price']} | "
            f"{row['s5_first_trigger_touch_utc']} | {row['s5_first_tp_touch_after_trigger_utc']} | "
            f"{row['realized_pl_jpy']} | {row['post_trigger_mae_pips']} | {row['post_trigger_mfe_pips']} |"
        )
    lines.extend(["", "## Slippage Sensitivity", ""])
    lines.extend([
        "| extra adverse pips | net JPY | expectancy JPY/trade | wins | losses | min trade JPY |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["slippage_sensitivity"]["scenarios"]:
        lines.append(
            f"| {row['total_extra_adverse_slippage_pips']} | {row['net_jpy']} | "
            f"{row['expectancy_jpy_per_trade']} | {row['wins']} | {row['losses']} | {row['min_trade_jpy']} |"
        )
    lines.extend(["", "## Invalidation", ""])
    lines.append(f"- Status: `{payload['invalidation_model']['status']}`")
    lines.append(
        "- Scout gap: "
        + payload["invalidation_model"]["scout_gap"]
    )
    lines.extend(["", "## Remaining Blockers", ""])
    for blocker in payload["remaining_blockers"]:
        lines.append(f"- `{blocker['code']}`: {blocker['status']} - {blocker['evidence']}")
    lines.extend(["", "## Validation", "", "```bash"])
    lines.append("python3 -m json.tool data/eurusd_short_breakout_failure_stop_harvest_replay.json >/dev/null")
    lines.append("PYTHONPATH=src python3 -m unittest tests.test_eurusd_stop_harvest_replay -v")
    lines.append("PYTHONPATH=src python3 -m unittest tests.test_live_runtime_sync -v")
    lines.append("```")
    return "\n".join(lines) + "\n"


def _full_price(tx: dict[str, Any]) -> dict[str, float | None]:
    full = tx.get("fullPrice") if isinstance(tx.get("fullPrice"), dict) else {}
    bids = full.get("bids") if isinstance(full.get("bids"), list) else []
    asks = full.get("asks") if isinstance(full.get("asks"), list) else []
    bid = _float((bids[0] or {}).get("price")) if bids else None
    ask = _float((asks[0] or {}).get("price")) if asks else None
    return {"bid": bid, "ask": ask}


def _spread_pips(full_price: dict[str, Any] | None) -> float | None:
    if not full_price:
        return None
    bid = full_price.get("bid")
    ask = full_price.get("ask")
    if bid is None or ask is None:
        return None
    return (float(ask) - float(bid)) / PIP


def _short_pips(entry: float | None, exit_price: float | None) -> float:
    if entry is None or exit_price is None:
        return 0.0
    return (entry - exit_price) / PIP


def _positive_pips(price_diff: float) -> float:
    return max(0.0, price_diff / PIP)


def _dist(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(clean)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {"min": _round(ordered[0]), "median": _round(median), "max": _round(ordered[-1])}


def _floor_s5(value: datetime) -> datetime:
    second = value.second - (value.second % S5_SECONDS)
    return value.replace(second=second, microsecond=0)


def _ceil_s5(value: datetime) -> datetime:
    floored = _floor_s5(value)
    if floored == value.replace(microsecond=0) and value.microsecond == 0:
        return floored
    return floored + timedelta(seconds=S5_SECONDS)


def _parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        tz = ""
        if "+" in tail:
            frac, tz = tail.split("+", 1)
            tz = "+" + tz
        elif "-" in tail:
            frac, tz = tail.split("-", 1)
            tz = "-" + tz
        else:
            frac = tail
        frac = "".join(ch for ch in frac if ch.isdigit())[:6].ljust(6, "0")
        text = f"{head}.{frac}{tz}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    if math.isinf(number):
        return number
    return round(number, digits)


def _price(value: Any) -> float | None:
    return _round(value, 5)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
