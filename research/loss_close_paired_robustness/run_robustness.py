#!/usr/bin/env python3
"""Preregistered, read-only robustness replay for SL hedge alternatives.

The script is intentionally research-local.  It reads the sealed execution
ledger and complete OANDA S5 bid/ask candles, never imports a broker write
client, never fills missing bars, and never opens the forward holdout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
from statistics import mean
from typing import Any, Iterable


CONTRACT = "loss_close_paired_robustness_result_v1"
ARMS = (
    "CONTROL_CURRENT_SL",
    "A_SL_REVERSE_STOP_025",
    "A_SL_REVERSE_STOP_035",
    "B_INITIAL_EQUAL_HEDGE",
    "B_SL_EQUAL_HEDGE",
)
WINDOWS = (
    ("INITIAL_16D", "2026-06-23T07:46:03.151624347Z", "2026-07-09T07:46:03.151624347Z"),
    ("DOUBLE_32D", "2026-06-07T07:46:03.151624347Z", "2026-07-09T07:46:03.151624347Z"),
    ("QUADRUPLE_64D", "2026-05-06T07:46:03.151624347Z", "2026-07-09T07:46:03.151624347Z"),
)
S5_SECONDS = 5
UNWIND_SECONDS = 3600
EMBARGO_SECONDS = 3600
MIN_EVENTS_PER_SPLIT = 30
MARGIN_RATE = 0.04
PAIR_MARGIN_CAP = 0.45
TOTAL_MARGIN_CAP = 0.92
RUIN_FLOOR_RATIO = 0.10
DAILY_DD_LIMIT_RATIO = 0.02
MARGIN_CLOSEOUT_RATIO = 0.50
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_809
PARTIAL_FILL_RATIOS = (1.0, 0.75, 0.5)
_FILE_RE = re.compile(r"_S5_BA_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)\.jsonl(?:\.gz)?$")


def parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # OANDA/SQLite preserve nanoseconds while datetime accepts microseconds.
    match = re.match(r"^(.*\.)(\d+)([+-]\d\d:\d\d)$", text)
    if match and len(match.group(2)) > 6:
        text = match.group(1) + match.group(2)[:6] + match.group(3)
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def floor_s5(value: datetime) -> datetime:
    return value.replace(microsecond=0) - timedelta(seconds=value.second % S5_SECONDS)


def file_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_events(path: Path) -> list[dict[str, Any]]:
    query = """
        WITH fills AS (
          SELECT trade_id,pair,side,ABS(units) units,price entry,ts_utc fill_at,order_id,event_uid fill_uid
          FROM execution_events WHERE event_type='ORDER_FILLED'
        ), accepted AS (
          SELECT order_id,tp,sl,event_uid accepted_uid
          FROM execution_events WHERE event_type='ORDER_ACCEPTED'
        ), closes AS (
          SELECT trade_id,event_uid close_uid,ts_utc close_at,price close_price,
                 realized_pl_jpy,financing_jpy,exit_reason
          FROM execution_events WHERE event_type='TRADE_CLOSED'
        )
        SELECT fills.*,accepted.tp,accepted.sl,accepted.accepted_uid,
               closes.close_uid,closes.close_at,closes.close_price,
               closes.realized_pl_jpy,closes.financing_jpy
        FROM fills JOIN accepted USING(order_id) JOIN closes USING(trade_id)
        WHERE closes.exit_reason='STOP_LOSS_ORDER'
          AND accepted.tp IS NOT NULL AND accepted.sl IS NOT NULL
        ORDER BY fills.fill_at
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(query)]
    finally:
        connection.close()


def index_sources(roots: Iterable[Path]) -> dict[str, list[tuple[datetime, datetime, Path]]]:
    result: dict[str, list[tuple[datetime, datetime, Path]]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("**/*_S5_BA_*.jsonl*"):
            match = _FILE_RE.search(path.name)
            if not match:
                continue
            pair = "_".join(path.name.split("_")[:2])
            result[pair].append((file_time(match.group(1)), file_time(match.group(2)), path))
    return result


def select_source(
    sources: Iterable[tuple[datetime, datetime, Path]], lo: datetime, hi: datetime
) -> Path | None:
    eligible = [item for item in sources if item[0] <= lo and item[1] >= hi]
    if not eligible:
        return None
    # Prefer the smallest containing artifact.  This selects the bounded
    # run-owned fetch over a multi-month file without changing its prices.
    return min(eligible, key=lambda item: ((item[1] - item[0]).total_seconds(), str(item[2])))[2]


def load_candles(path: Path, lo: datetime, hi: datetime) -> dict[datetime, dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[datetime, dict[str, Any]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ts = parse_time(str(row["time"]))
            if ts < lo or ts > hi:
                continue
            if row.get("complete") is not True or not isinstance(row.get("bid"), dict) or not isinstance(row.get("ask"), dict):
                continue
            rows[ts] = row
    return rows


def touch(event: dict[str, Any], candles: dict[datetime, dict[str, Any]]) -> tuple[datetime, str] | None:
    start = floor_s5(parse_time(event["fill_at"]))
    end = floor_s5(parse_time(event["close_at"])) + timedelta(seconds=S5_SECONDS)
    side, tp, sl = event["side"], float(event["tp"]), float(event["sl"])
    for ts in sorted(key for key in candles if start <= key <= end):
        row = candles[ts]
        if side == "LONG":
            tp_hit = float(row["bid"]["h"]) >= tp
            sl_hit = float(row["bid"]["l"]) <= sl
        else:
            tp_hit = float(row["ask"]["l"]) <= tp
            sl_hit = float(row["ask"]["h"]) >= sl
        if tp_hit or sl_hit:
            return ts, "AMBIGUOUS" if tp_hit and sl_hit else "TP" if tp_hit else "SL"
    return None


def gap_count(candles: dict[datetime, dict[str, Any]], lo: datetime, hi: datetime) -> int:
    expected = int((hi - lo).total_seconds() // S5_SECONDS) + 1
    observed = sum(lo <= ts <= hi for ts in candles)
    return max(0, expected - observed)


def implied_quote_to_jpy(event: dict[str, Any]) -> float | None:
    if str(event["pair"]).endswith("_JPY"):
        return 1.0
    direction = 1.0 if event["side"] == "LONG" else -1.0
    price_pnl = direction * (float(event["close_price"]) - float(event["entry"])) * int(event["units"])
    realized_price_pnl = float(event["realized_pl_jpy"] or 0.0) - float(event["financing_jpy"] or 0.0)
    if price_pnl == 0.0:
        return None
    value = realized_price_pnl / price_pnl
    return value if math.isfinite(value) and value > 0.0 else None


def spread(row: dict[str, Any]) -> float:
    return max(0.0, float(row["ask"]["c"]) - float(row["bid"]["c"]))


def mid(row: dict[str, Any]) -> float:
    return (float(row["bid"]["c"]) + float(row["ask"]["c"])) / 2.0


def crosses_rollover(start: datetime, end: datetime) -> bool:
    cursor = start.replace(hour=21, minute=0, second=0, microsecond=0)
    if cursor < start:
        cursor += timedelta(days=1)
    return cursor <= end


def pnl(direction: float, entry: float, exit_price: float, units: float, quote_to_jpy: float) -> float:
    return direction * (exit_price - entry) * units * quote_to_jpy


def event_result(event: dict[str, Any], source: Path | None, candles: dict[datetime, dict[str, Any]]) -> dict[str, Any]:
    identity = {
        "trade_id": str(event["trade_id"]),
        "pair": event["pair"],
        "side": event["side"],
        "units": int(event["units"]),
        "fill_at_utc": event["fill_at"],
        "close_at_utc": event["close_at"],
        "source": str(source) if source else None,
    }
    blockers: list[str] = []
    if source is None:
        blockers.append("NO_CONTAINING_S5_SOURCE")
    quote_to_jpy = implied_quote_to_jpy(event)
    if quote_to_jpy is None:
        blockers.append("QUOTE_TO_JPY_UNRESOLVED")
    first = touch(event, candles) if candles else None
    if first is None:
        blockers.append("NO_S5_PROTECTION_FIRST_TOUCH")
    elif first[1] != "SL":
        blockers.append(f"FIRST_TOUCH_NOT_UNAMBIGUOUS_SL:{first[1]}")
    if blockers:
        return {**identity, "status": "BLOCKED", "blockers": blockers, "arms": {}}

    assert first is not None and quote_to_jpy is not None
    trigger = first[0]
    unwind = trigger + timedelta(seconds=UNWIND_SECONDS)
    entry_ts = floor_s5(parse_time(event["fill_at"]))
    close_ts = floor_s5(parse_time(event["close_at"]))
    entry_row, trigger_row = candles.get(entry_ts), candles.get(trigger)
    close_row, unwind_row = candles.get(close_ts), candles.get(unwind)
    for label, row in (("ENTRY", entry_row), ("TRIGGER", trigger_row), ("CONTROL_CLOSE", close_row), ("UNWIND", unwind_row)):
        if row is None:
            blockers.append(f"{label}_S5_CANDLE_MISSING")
    if blockers:
        return {
            **identity,
            "status": "BLOCKED",
            "blockers": blockers,
            "baseline_sl_trigger_utc": iso(trigger),
            "arms": {},
        }
    assert entry_row and trigger_row and close_row and unwind_row

    side = str(event["side"])
    direction = 1.0 if side == "LONG" else -1.0
    hedge_direction = -direction
    units = float(event["units"])
    control_net = float(event["realized_pl_jpy"] or 0.0) + float(event["financing_jpy"] or 0.0)
    adverse_price_slip = max(
        0.0,
        float(event["sl"]) - float(event["close_price"])
        if side == "LONG"
        else float(event["close_price"]) - float(event["sl"]),
    )
    adverse_slippage_jpy = adverse_price_slip * units * quote_to_jpy
    trigger_spread = spread(trigger_row)
    unwind_spread = spread(unwind_row)
    entry_spread = spread(entry_row)
    close_spread = spread(close_row)
    control_margin = units * mid(entry_row) * quote_to_jpy * MARGIN_RATE
    full_gap_count = gap_count(candles, entry_ts, unwind)
    rollover_unknown = crosses_rollover(trigger, unwind)

    trigger_exec = float(event["close_price"])
    reverse_exit = float(unwind_row["ask"]["c"] if side == "LONG" else unwind_row["bid"]["c"])
    original_unwind = float(unwind_row["bid"]["c"] if side == "LONG" else unwind_row["ask"]["c"])
    hedge_unwind = reverse_exit
    opposite_entry_initial = float(event["entry"]) - entry_spread if side == "LONG" else float(event["entry"]) + entry_spread
    opposite_exit_control = float(close_row["ask"]["c"] if side == "LONG" else close_row["bid"]["c"])

    arms: dict[str, dict[str, Any]] = {}
    arms["CONTROL_CURRENT_SL"] = arm_row(
        net=control_net,
        control=control_net,
        spread_cost=0.0,
        slippage_cost=adverse_slippage_jpy,
        margin=control_margin,
        gross_margin=control_margin,
        margin_hours=control_margin * max(0.0, (parse_time(event["close_at"]) - parse_time(event["fill_at"])).total_seconds()) / 3600.0,
        gaps=full_gap_count,
        fill_order="ACTUAL_LEDGER_FILL",
        dual_unwind="NOT_APPLICABLE",
        partial_nets={"1.0": control_net},
        trend=False,
        mean_reversion_failure=False,
        financing_unknown=False,
    )

    for scale, arm_id in ((0.25, "A_SL_REVERSE_STOP_025"), (0.35, "A_SL_REVERSE_STOP_035")):
        raw_hedge = pnl(hedge_direction, trigger_exec, reverse_exit, units * scale, quote_to_jpy)
        slippage_cost = adverse_slippage_jpy * scale * 2.0
        net = control_net + raw_hedge - slippage_cost
        partial = {
            str(ratio): control_net + raw_hedge * ratio - slippage_cost * ratio
            for ratio in PARTIAL_FILL_RATIOS
        }
        gross_margin = units * mid(trigger_row) * quote_to_jpy * MARGIN_RATE * (1.0 + scale)
        arms[arm_id] = arm_row(
            net=net,
            control=control_net,
            spread_cost=(trigger_spread + unwind_spread) * units * scale * quote_to_jpy,
            slippage_cost=slippage_cost,
            margin=max(control_margin, control_margin * scale),
            gross_margin=max(control_margin, gross_margin),
            margin_hours=control_margin * scale,
            gaps=full_gap_count,
            fill_order="UNRESOLVED_SAME_S5_SL_CLOSE_AND_REVERSE_OPEN",
            dual_unwind="SINGLE_REVERSE_LEG",
            partial_nets=partial,
            trend=raw_hedge > 0.0,
            mean_reversion_failure=raw_hedge <= 0.0,
            financing_unknown=rollover_unknown,
        )

    initial_hedge = pnl(hedge_direction, opposite_entry_initial, opposite_exit_control, units, quote_to_jpy)
    initial_slippage = adverse_slippage_jpy * 2.0
    initial_net = control_net + initial_hedge - initial_slippage
    arms["B_INITIAL_EQUAL_HEDGE"] = arm_row(
        net=initial_net,
        control=control_net,
        spread_cost=(entry_spread + close_spread) * units * quote_to_jpy,
        slippage_cost=initial_slippage,
        margin=control_margin,
        gross_margin=control_margin * 2.0,
        margin_hours=control_margin * (parse_time(event["close_at"]) - parse_time(event["fill_at"])).total_seconds() / 3600.0,
        gaps=gap_count(candles, entry_ts, close_ts),
        fill_order="SIMULTANEOUS_INITIAL_HEDGE_ENTRY_S5_UNRESOLVED",
        dual_unwind="UNRESOLVED_SAME_S5_DUAL_UNWIND",
        partial_nets={
            str(ratio): control_net + initial_hedge * ratio - initial_slippage * ratio
            for ratio in PARTIAL_FILL_RATIOS
        },
        trend=False,
        mean_reversion_failure=False,
        financing_unknown=crosses_rollover(parse_time(event["fill_at"]), parse_time(event["close_at"])),
    )

    original_raw = pnl(direction, float(event["entry"]), original_unwind, units, quote_to_jpy)
    sl_hedge_raw = pnl(hedge_direction, trigger_exec, hedge_unwind, units, quote_to_jpy)
    sl_hedge_slippage = adverse_slippage_jpy * 3.0
    sl_hedge_net = original_raw + sl_hedge_raw - sl_hedge_slippage + float(event["financing_jpy"] or 0.0)
    arms["B_SL_EQUAL_HEDGE"] = arm_row(
        net=sl_hedge_net,
        control=control_net,
        spread_cost=(trigger_spread + 2.0 * unwind_spread) * units * quote_to_jpy,
        slippage_cost=sl_hedge_slippage,
        margin=control_margin,
        gross_margin=control_margin * 2.0,
        margin_hours=control_margin,
        gaps=full_gap_count,
        fill_order="UNRESOLVED_SAME_S5_SL_TRIGGER_AND_HEDGE_OPEN",
        dual_unwind="UNRESOLVED_SAME_S5_DUAL_UNWIND",
        partial_nets={
            str(ratio): original_raw + sl_hedge_raw * ratio - adverse_slippage_jpy * (1.0 + 2.0 * ratio)
            for ratio in PARTIAL_FILL_RATIOS
        },
        trend=False,
        mean_reversion_failure=False,
        financing_unknown=rollover_unknown,
    )
    return {
        **identity,
        "status": "CALCULATED_DIAGNOSTIC",
        "blockers": [],
        "baseline_sl_trigger_utc": iso(trigger),
        "fixed_unwind_utc": iso(unwind),
        "quote_to_jpy": quote_to_jpy,
        "s5_gap_count_entry_to_unwind": full_gap_count,
        "arms": arms,
    }


def arm_row(
    *, net: float, control: float, spread_cost: float, slippage_cost: float,
    margin: float, gross_margin: float, margin_hours: float, gaps: int,
    fill_order: str, dual_unwind: str, partial_nets: dict[str, float],
    trend: bool, mean_reversion_failure: bool, financing_unknown: bool,
) -> dict[str, Any]:
    return {
        "net_jpy": net,
        "paired_delta_jpy": net - control,
        "cost_breakdown_jpy": {
            "intrinsic_spread_estimate": spread_cost,
            "explicit_fee": 0.0,
            "slippage_stress": slippage_cost,
            "financing_known": not financing_unknown,
        },
        "peak_broker_margin_jpy": margin,
        "peak_double_gross_margin_jpy": gross_margin,
        "incremental_margin_hours_jpy": margin_hours,
        "path_gap_count": gaps,
        "fill_order_status": fill_order,
        "dual_unwind_status": dual_unwind,
        "partial_fill_net_jpy": partial_nets,
        "trend_continuation": trend,
        "mean_reversion_failure": mean_reversion_failure,
    }


def assign_split(rows: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: parse_time(row["fill_at_utc"]))
    if not ordered:
        return {}
    cut = max(1, math.floor(len(ordered) * 0.60))
    train = ordered[:cut]
    train_end = parse_time(train[-1]["fill_at_utc"])
    result = {row["trade_id"]: "TRAIN" for row in train}
    for row in ordered[cut:]:
        result[row["trade_id"]] = (
            "EMBARGO" if parse_time(row["fill_at_utc"]) <= train_end + timedelta(seconds=EMBARGO_SECONDS)
            else "VALIDATION"
        )
    return result


def profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    if losses == 0.0:
        return math.inf if gains > 0.0 else None
    return gains / losses


def max_drawdown(values: list[float]) -> float:
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def bootstrap_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(mean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAP_SAMPLES))
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def aggregate(rows: list[dict[str, Any]], arm_id: str, split: str, total_events: int) -> dict[str, Any]:
    selected = [row for row in rows if row.get("split") == split and arm_id in row.get("arms", {})]
    values = [float(row["arms"][arm_id]["net_jpy"]) for row in selected]
    deltas = [float(row["arms"][arm_id]["paired_delta_jpy"]) for row in selected]
    lo, hi = bootstrap_ci(deltas)
    peak_control_margin = max(
        (float(row["arms"]["CONTROL_CURRENT_SL"]["peak_broker_margin_jpy"]) for row in selected),
        default=0.0,
    )
    initial_equity = peak_control_margin / PAIR_MARGIN_CAP if peak_control_margin else 0.0
    peak_margin = max((float(row["arms"][arm_id]["peak_broker_margin_jpy"]) for row in selected), default=0.0)
    peak_gross = max((float(row["arms"][arm_id]["peak_double_gross_margin_jpy"]) for row in selected), default=0.0)
    dd = max_drawdown(values)
    path_complete = all(int(row["arms"][arm_id]["path_gap_count"]) == 0 for row in selected) if selected else False
    fill_resolved = all("UNRESOLVED" not in str(row["arms"][arm_id]["fill_order_status"]) for row in selected) if selected else False
    unwind_resolved = all("UNRESOLVED" not in str(row["arms"][arm_id]["dual_unwind_status"]) for row in selected) if selected else False
    financing_known = all(bool(row["arms"][arm_id]["cost_breakdown_jpy"]["financing_known"]) for row in selected) if selected else False
    partial_positive = all(
        min(float(value) for value in row["arms"][arm_id]["partial_fill_net_jpy"].values()) > 0.0
        for row in selected
    ) if selected else False
    net = sum(values)
    pf = profit_factor(values)
    gates = {
        "minimum_events": len(selected) >= MIN_EVENTS_PER_SPLIT,
        "net_positive": net > 0.0,
        "profit_factor_above_one": pf is not None and pf > 1.0,
        "paired_lcb_positive": lo is not None and lo > 0.0,
        "path_complete": path_complete,
        "fill_order_resolved": fill_resolved,
        "dual_unwind_complete": unwind_resolved,
        "financing_complete": financing_known,
        "partial_fill_stress_positive": partial_positive,
        "drawdown_within_daily_risk": bool(initial_equity) and dd <= initial_equity * DAILY_DD_LIMIT_RATIO,
        "margin_within_cap": bool(initial_equity) and peak_gross <= initial_equity * TOTAL_MARGIN_CAP,
        "no_ruin_floor_breach_proxy": bool(initial_equity) and -sum(values) < initial_equity * (1.0 - RUIN_FLOOR_RATIO),
        "no_margin_closeout_proxy": bool(initial_equity) and initial_equity + min(0.0, sum(values)) > peak_margin * MARGIN_CLOSEOUT_RATIO,
    }
    costs = {
        key: sum(float(row["arms"][arm_id]["cost_breakdown_jpy"][key]) for row in selected)
        for key in ("intrinsic_spread_estimate", "explicit_fee", "slippage_stress")
    }
    return {
        "split": split,
        "arm": arm_id,
        "trades": len(selected),
        "coverage": len(selected) / total_events if total_events else 0.0,
        "net_jpy": net,
        "profit_factor": pf,
        "expectancy_jpy": mean(values) if values else None,
        "max_drawdown_jpy": dd,
        "initial_equity_proxy_jpy": initial_equity,
        "peak_broker_margin_jpy": peak_margin,
        "peak_double_gross_margin_jpy": peak_gross,
        "gross_margin_utilization_pct": 100.0 * peak_gross / initial_equity if initial_equity else None,
        "ruin_floor_breach_proxy": not gates["no_ruin_floor_breach_proxy"],
        "margin_closeout_proxy": not gates["no_margin_closeout_proxy"],
        "cost_breakdown_jpy": costs,
        "paired_delta_mean_jpy": mean(deltas) if deltas else None,
        "paired_bootstrap_95pct_ci": [lo, hi],
        "paired_lcb_jpy": lo,
        "trend_continuation_count": sum(bool(row["arms"][arm_id]["trend_continuation"]) for row in selected),
        "mean_reversion_failure_count": sum(bool(row["arms"][arm_id]["mean_reversion_failure"]) for row in selected),
        "gates": gates,
        "accepted": arm_id != "CONTROL_CURRENT_SL" and all(gates.values()),
    }


def run(repo: Path) -> dict[str, Any]:
    prereg = repo / "research/loss_close_paired_robustness/preregister_v1.json"
    ledger = repo / "data/execution_ledger.db"
    frozen = json.loads(prereg.read_text())
    if sha256_file(ledger) != frozen["source_bindings"]["execution_ledger_sha256"]:
        raise RuntimeError("execution ledger changed after preregistration")
    roots = (
        repo / "logs/replay/oanda_history",
        repo / "logs/replay/oanda_prediction_truth",
        repo / "research/loss_close_paired_robustness/s5_cache",
    )
    source_index = index_sources(roots)
    events = load_events(ledger)
    results: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for event in events:
        lo = floor_s5(parse_time(event["fill_at"]))
        hi = floor_s5(parse_time(event["close_at"])) + timedelta(seconds=UNWIND_SECONDS + S5_SECONDS)
        source = select_source(source_index.get(str(event["pair"]), ()), lo, hi)
        candles = load_candles(source, lo, hi) if source else {}
        if source:
            source_hashes[str(source.relative_to(repo))] = sha256_file(source)
        results.append(event_result(event, source.relative_to(repo) if source else None, candles))

    windows = []
    for window_id, start_text, end_text in WINDOWS:
        start, end = parse_time(start_text), parse_time(end_text)
        window_rows = [row for row in results if start <= parse_time(row["fill_at_utc"]) <= end]
        split_map = assign_split(window_rows)
        for row in window_rows:
            row.setdefault("window_splits", {})[window_id] = split_map.get(row["trade_id"])
        scoped = [{**row, "split": split_map.get(row["trade_id"])} for row in window_rows]
        calculated = [row for row in scoped if row["status"] == "CALCULATED_DIAGNOSTIC"]
        by_arm = {
            arm: {
                split: aggregate(calculated, arm, split, len(window_rows))
                for split in ("TRAIN", "VALIDATION")
            }
            for arm in ARMS
        }
        arm_decisions = {
            arm: "ACCEPT" if by_arm[arm]["VALIDATION"]["accepted"] else "REJECT"
            for arm in ARMS if arm != "CONTROL_CURRENT_SL"
        }
        windows.append({
            "id": window_id,
            "from_utc": start_text,
            "to_utc": end_text,
            "cohort_events": len(window_rows),
            "calculated_events": len(calculated),
            "blocked_events": len(window_rows) - len(calculated),
            "split_counts": {
                split: sum(value == split for value in split_map.values())
                for split in ("TRAIN", "EMBARGO", "VALIDATION")
            },
            "arms": by_arm,
            "arm_decisions": arm_decisions,
        })

    final_decisions = {
        arm: "ACCEPT" if all(window["arm_decisions"][arm] == "ACCEPT" for window in windows) else "REJECT"
        for arm in ARMS if arm != "CONTROL_CURRENT_SL"
    }
    return {
        "contract": CONTRACT,
        "preregister_sha256": sha256_file(prereg),
        "ledger_sha256": sha256_file(ledger),
        "holdout_used": False,
        "permissions": {"read_only": True, "paper": False, "live": False, "broker_order": False, "deploy": False},
        "account_contract": {"currency": "JPY", "hedging_enabled": True, "margin_rate": MARGIN_RATE},
        "source_hashes": dict(sorted(source_hashes.items())),
        "events": results,
        "windows": windows,
        "final_decisions": final_decisions,
        "overall_decision": "ACCEPT" if all(value == "ACCEPT" for value in final_decisions.values()) else "REJECT",
        "proof_limitations": [
            "STOP-loss-trigger cohort does not establish entry-strategy profitability",
            "S5 cannot resolve ordering of same-candle close/open or dual-unwind fills",
            "missing OANDA S5 intervals are not interpolated and make drawdown proof incomplete",
            "ruin is a deterministic floor proxy, not an estimated probability",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    report = run(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "overall_decision": report["overall_decision"],
        "final_decisions": report["final_decisions"],
        "events": len(report["events"]),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
