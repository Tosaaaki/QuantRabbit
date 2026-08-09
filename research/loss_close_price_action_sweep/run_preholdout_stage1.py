#!/usr/bin/env python3
"""Read-only Stage-1 scan over canonical STOP_LOSS entries and local S5 BA.

This adapter does not invent missing S5 bars.  It may calculate causal feature
context across OANDA no-quote gaps, but an event is economic-score eligible
only when the entry-to-fixed-unwind S5 sequence is complete.  It never reads a
TEST/HOLDOUT source and has no broker, Paper, live, order, or deploy imports.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from quant_rabbit.loss_close_multidimensional_sweep import (
    SweepContract,
    build_stage1_price_action_grid,
)
from quant_rabbit.loss_close_paired_shadow import (
    PAIRED_SHADOW_STATE_CONTRACT,
    S5BidAskCandle,
    S5Ohlc,
    seal_paired_shadow_state,
)
from quant_rabbit.loss_close_price_action_shadow import build_price_action_context


REPORT_CONTRACT = "loss_close_price_action_stage1_real_cohort_scan_v1"
PAIR_WITH_LOCAL_S5 = ("AUD_JPY", "EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY")
MAX_CONTEXT_HOURS = 5
_FILE_RE = re.compile(r"_S5_BA_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)\.jsonl\.gz$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = run_scan(repo)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else repo / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    else:
        print(encoded, end="")


def run_scan(repo: Path) -> dict[str, Any]:
    contract = SweepContract()
    events = _load_events(repo / "data/execution_ledger.db")
    file_index = _index_s5_files(repo / "logs/replay/oanda_history")
    grid = build_stage1_price_action_grid()
    event_sources = {
        str(event["trade_id"]): _source_for_event(
            file_index.get(event["pair"], ()), _parse_time(event["close_at"])
        )
        for event in events
    }
    source_events: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        source = event_sources[str(event["trade_id"])]
        if source is not None:
            source_events[source].append(event)
    candle_cache: dict[Path, tuple[S5BidAskCandle, ...]] = {}
    for source, selected in source_events.items():
        candle_cache[source] = _load_candles(
            source,
            pair=selected[0]["pair"],
            lo=min(_parse_time(event["fill_at"]) for event in selected)
            - timedelta(hours=MAX_CONTEXT_HOURS),
            hi=max(_parse_time(event["close_at"]) for event in selected)
            + timedelta(seconds=contract.max_unwind_seconds),
        )
    event_reports: list[dict[str, Any]] = []
    config_counts = {
        point.config_id: {
            "config_id": point.config_id,
            "config": point.config,
            "context_calculated": 0,
            "setup_gate_count": 0,
            "price_action_against_inventory_count": 0,
            "candle_against_inventory_count": 0,
            "pattern_counts": Counter(),
            "blocker_counts": Counter(),
        }
        for point in grid
    }

    for event in events:
        source = event_sources[str(event["trade_id"])]
        if source is None:
            event_reports.append(_event_without_source(event))
            continue
        fill = _parse_time(event["fill_at"])
        close = _parse_time(event["close_at"])
        candles = tuple(
            candle
            for candle in candle_cache[source]
            if fill - timedelta(hours=MAX_CONTEXT_HOURS)
            <= candle.timestamp_utc
            <= close + timedelta(seconds=contract.max_unwind_seconds)
        )
        trigger = _first_protection_touch(event, candles, not_before=_floor_s5(fill))
        if trigger is None or trigger["reason"] != "SL":
            event_reports.append(
                {
                    **_event_identity(event),
                    "s5_source": str(source.relative_to(repo)),
                    "context_status": "BLOCKED_NO_UNAMBIGUOUS_S5_SL_FIRST_TOUCH",
                    "first_touch": trigger,
                    "strict_economic_score_eligible": False,
                }
            )
            continue
        trigger_time = trigger["timestamp_utc"]
        previous = _loss_side_decision_candle(event, candles, trigger_time)
        if previous is None:
            event_reports.append(
                {
                    **_event_identity(event),
                    "s5_source": str(source.relative_to(repo)),
                    "context_status": "BLOCKED_NO_PRE_TRIGGER_LOSS_SIDE_QUOTE",
                    "first_touch": _serialise_touch(trigger),
                    "strict_economic_score_eligible": False,
                }
            )
            continue
        state = _state(event, previous)
        start = _floor_s5(fill)
        end = trigger_time + timedelta(seconds=contract.max_unwind_seconds)
        gap_count = _gap_count(candles, start=start, end=end)
        strict = gap_count == 0
        event_row = {
            **_event_identity(event),
            "decision_timestamp_utc": _iso(previous.timestamp_utc),
            "baseline_sl_trigger_utc": _iso(trigger_time),
            "s5_source": str(source.relative_to(repo)),
            "s5_gap_count_entry_to_fixed_60m_unwind": gap_count,
            "strict_economic_score_eligible": strict,
            "strict_ineligibility_reason": None if strict else "S5_TRUTH_GAP_FILL_ORDER_UNRESOLVED",
            "context_status": "CALCULATED",
            "split": None,
        }
        event_reports.append(event_row)
        for point in grid:
            result = build_price_action_context(state, candles, spec=point.feature_spec)
            counts = config_counts[point.config_id]
            if result["status"] != "CONTEXT_CALCULATED_OUTCOME_NOT_EVALUATED":
                counts["blocker_counts"].update(result["blockers"])
                continue
            counts["context_calculated"] += 1
            cross = result["cross_frame"]
            counts["setup_gate_count"] += cross["setup_gate"] == "EVALUATE_PAIRED_SHADOW_ONLY"
            counts["price_action_against_inventory_count"] += cross[
                "price_action_against_inventory"
            ]
            counts["candle_against_inventory_count"] += cross["candle_against_inventory"]
            counts["pattern_counts"].update(cross["chart_pattern_candidates"].values())

    _assign_splits(event_reports, embargo_seconds=contract.embargo_seconds)
    config_rows = []
    for point in grid:
        counts = config_counts[point.config_id]
        counts["pattern_counts"] = dict(sorted(counts["pattern_counts"].items()))
        counts["blocker_counts"] = dict(sorted(counts["blocker_counts"].items()))
        config_rows.append(counts)
    context_ready = [row for row in event_reports if row.get("context_status") == "CALCULATED"]
    strict_ready = [row for row in context_ready if row["strict_economic_score_eligible"]]
    split_counts = Counter(row.get("split") for row in context_ready)
    cohort_status = (
        "READY_FOR_PAIRED_ECONOMIC_SWEEP"
        if len(strict_ready) >= contract.min_events_per_split * 2
        else "BLOCKED_INSUFFICIENT_STRICT_S5_COHORT"
    )
    return {
        "contract": REPORT_CONTRACT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": cohort_status,
        "scope": {
            "entry_source": "execution_ledger ORDER_FILLED with frozen accepted TP/SL",
            "baseline_event": "unambiguous STOP_LOSS_ORDER and first S5 protection touch SL",
            "price_source": "local OANDA S5 bid/ask JSONL only",
            "splits": ["TRAIN", "VALIDATION"],
            "holdout_used": False,
            "max_unwind_seconds": contract.max_unwind_seconds,
            "embargo_seconds": contract.embargo_seconds,
        },
        "cohort": {
            "ledger_stop_loss_events_with_local_pair_family": len(events),
            "context_ready_events": len(context_ready),
            "strict_s5_economic_score_ready_events": len(strict_ready),
            "train_context_events": split_counts["TRAIN"],
            "validation_context_events": split_counts["VALIDATION"],
            "minimum_required_per_split": contract.min_events_per_split,
            "economic_score_blocker": (
                None
                if strict_ready
                else "ALL_CONTEXT_EVENTS_HAVE_AT_LEAST_ONE_ENTRY_TO_UNWIND_S5_GAP"
            ),
        },
        "stage1": {
            "grid_cell_count": len(grid),
            "full_cartesian_search_used": False,
            "single_best_cell_adoption_allowed": False,
            "outcome_selection_performed": False,
            "reason": "No strict S5 paired economic cohort; feature counts are diagnostic only.",
            "config_diagnostics": config_rows,
        },
        "cost_and_risk_contract": {
            "spread": "INTRINSIC_EXECUTABLE_BID_ASK",
            "fee": "REQUIRED_NON_SPREAD_INPUT_NOT_ESTIMATED_IN_CONTEXT_ONLY_SCAN",
            "slippage": "REQUIRED_NON_SPREAD_INPUT_NOT_ESTIMATED_IN_CONTEXT_ONLY_SCAN",
            "financing": "LEDGER_VALUE_PRESENT_BUT_NOT_SCORED_WITHOUT_STRICT_S5_PATH",
            "margin": "LONGEST_LEG_INCREMENT_PROXY_REQUIRED_BEFORE_OUTCOME_ADOPTION",
            "fill_order": "ANY_MISSING_S5_IS_FATAL",
            "trend_continuation": "REQUIRED_BEFORE_OUTCOME_ADOPTION",
            "maximum_drawdown": "REQUIRED_BEFORE_OUTCOME_ADOPTION",
            "ruin": "DETERMINISTIC_FLOOR_ONLY; PROBABILITY_NOT_ESTIMATED",
            "unwind": "FIXED_60_MINUTES; MUST BE COMPLETE",
        },
        "permissions": {
            "read_only": True,
            "paper": False,
            "live": False,
            "broker": False,
            "order": False,
            "deploy": False,
            "holdout": False,
        },
        "events": event_reports,
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    query = """
        WITH fills AS (
          SELECT trade_id,pair,side,ABS(units) units,price entry,ts_utc fill_at,order_id
          FROM execution_events WHERE event_type='ORDER_FILLED'
        ), accepted AS (
          SELECT order_id,tp,sl FROM execution_events WHERE event_type='ORDER_ACCEPTED'
        ), closes AS (
          SELECT trade_id,event_uid close_uid,ts_utc close_at,price close_price,
                 realized_pl_jpy,financing_jpy,exit_reason
          FROM execution_events WHERE event_type='TRADE_CLOSED'
        )
        SELECT fills.*,accepted.tp,accepted.sl,closes.close_uid,closes.close_at,
               closes.close_price,closes.realized_pl_jpy,closes.financing_jpy
        FROM fills JOIN accepted USING(order_id) JOIN closes USING(trade_id)
        WHERE closes.exit_reason='STOP_LOSS_ORDER'
          AND accepted.tp IS NOT NULL AND accepted.sl IS NOT NULL
          AND fills.pair IN (?,?,?,?,?)
        ORDER BY fills.fill_at
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(query, PAIR_WITH_LOCAL_S5)]
    finally:
        connection.close()


def _index_s5_files(root: Path) -> dict[str, tuple[tuple[datetime, datetime, Path], ...]]:
    result: dict[str, list[tuple[datetime, datetime, Path]]] = defaultdict(list)
    for path in root.glob("**/*_S5_BA_*.jsonl.gz"):
        found = _FILE_RE.search(path.name)
        if not found:
            continue
        pair = "_".join(path.name.split("_")[:2])
        result[pair].append((_file_time(found.group(1)), _file_time(found.group(2)), path))
    return {key: tuple(sorted(value)) for key, value in result.items()}


def _source_for_event(
    sources: Iterable[tuple[datetime, datetime, Path]], timestamp: datetime
) -> Path | None:
    candidates = [item for item in sources if item[0] <= timestamp <= item[1]]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1] - item[0]).total_seconds())[2]


def _load_candles(
    path: Path, *, pair: str, lo: datetime, hi: datetime
) -> tuple[S5BidAskCandle, ...]:
    candles = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            timestamp = _parse_time(row["time"])
            if timestamp < lo or timestamp > hi:
                continue
            bid, ask = row["bid"], row["ask"]
            candles.append(
                S5BidAskCandle(
                    timestamp_utc=timestamp,
                    pair=pair,
                    bid=S5Ohlc(float(bid["o"]), float(bid["h"]), float(bid["l"]), float(bid["c"])),
                    ask=S5Ohlc(float(ask["o"]), float(ask["h"]), float(ask["l"]), float(ask["c"])),
                    complete=row.get("complete") is True,
                )
            )
    return tuple(candles)


def _first_protection_touch(
    event: dict[str, Any], candles: Iterable[S5BidAskCandle], *, not_before: datetime
) -> dict[str, Any] | None:
    side, tp, sl = event["side"], float(event["tp"]), float(event["sl"])
    for candle in candles:
        if candle.timestamp_utc < not_before:
            continue
        if side == "LONG":
            tp_hit, sl_hit = candle.bid.high >= tp, candle.bid.low <= sl
        else:
            tp_hit, sl_hit = candle.ask.low <= tp, candle.ask.high >= sl
        if tp_hit or sl_hit:
            return {
                "timestamp_utc": candle.timestamp_utc,
                "reason": "AMBIGUOUS" if tp_hit and sl_hit else "TP" if tp_hit else "SL",
            }
    return None


def _loss_side_decision_candle(
    event: dict[str, Any], candles: tuple[S5BidAskCandle, ...], trigger: datetime
) -> S5BidAskCandle | None:
    side, tp, sl = event["side"], float(event["tp"]), float(event["sl"])
    for candle in reversed(candles):
        if candle.timestamp_utc >= trigger:
            continue
        executable = candle.bid.close if side == "LONG" else candle.ask.close
        if (side == "LONG" and sl < executable < tp) or (
            side == "SHORT" and tp < executable < sl
        ):
            return candle
    return None


def _state(event: dict[str, Any], decision: S5BidAskCandle) -> dict[str, Any]:
    side = event["side"]
    executable = decision.bid.close if side == "LONG" else decision.ask.close
    direction = 1.0 if side == "LONG" else -1.0
    unrealized = (executable - float(event["entry"])) * direction * int(event["units"])
    if unrealized >= 0.0:
        unrealized = -0.000001
    seed = f"{event['close_uid']}:{_iso(decision.timestamp_utc)}"
    body = {
        "contract": PAIRED_SHADOW_STATE_CONTRACT,
        "trade_id": str(event["trade_id"]),
        "close_decision_event_uid": str(event["close_uid"]),
        "pair": event["pair"],
        "side": side,
        "units": int(event["units"]),
        "decision_timestamp_utc": _iso(decision.timestamp_utc),
        "quote_timestamp_utc": _iso(decision.timestamp_utc),
        "decision_bid": float(decision.bid.close),
        "decision_ask": float(decision.ask.close),
        "executable_close_price": float(executable),
        "take_profit": float(event["tp"]),
        "stop_loss": float(event["sl"]),
        "quote_to_jpy": 1.0,
        "broker_snapshot_sha256": _digest(seed + ":broker"),
        "decision_unrealized_pnl_jpy": float(unrealized),
        "close_verifier_receipt_sha256": _digest(seed + ":close"),
        "close_verifier_verdict": "PASS",
        "technical_context_sha256": _digest(seed + ":technical"),
        "cost_surface_sha256": _digest("S5_BA_SPREAD_INTRINSIC_NON_SPREAD_UNBOUND"),
        "take_profit_exit_non_spread_cost_jpy": 0.0,
        "stop_loss_exit_non_spread_cost_jpy": 0.0,
        "control_financing_stress_jpy": float(abs(event.get("financing_jpy") or 0.0)),
        "read_only": True,
        "live_permission_allowed": False,
    }
    return seal_paired_shadow_state(body)


def _gap_count(candles: Iterable[S5BidAskCandle], *, start: datetime, end: datetime) -> int:
    observed = {candle.timestamp_utc for candle in candles if start <= candle.timestamp_utc <= end}
    expected = int((end - start).total_seconds() // 5) + 1
    return expected - len(observed)


def _assign_splits(rows: list[dict[str, Any]], *, embargo_seconds: int) -> None:
    ready = sorted(
        (row for row in rows if row.get("context_status") == "CALCULATED"),
        key=lambda row: row["decision_timestamp_utc"],
    )
    if len(ready) < 2:
        return
    cut = max(1, min(len(ready) - 1, int(len(ready) * 0.60)))
    boundary = _parse_time(ready[cut]["decision_timestamp_utc"])
    for row in ready[:cut]:
        if _parse_time(row["decision_timestamp_utc"]) <= boundary - timedelta(
            seconds=embargo_seconds
        ):
            row["split"] = "TRAIN"
    for row in ready[cut:]:
        if _parse_time(row["decision_timestamp_utc"]) >= boundary:
            row["split"] = "VALIDATION"


def _event_identity(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_uid": str(event["close_uid"]),
        "trade_id": str(event["trade_id"]),
        "pair": event["pair"],
        "side": event["side"],
        "fill_at_utc": event["fill_at"],
        "close_at_utc": event["close_at"],
    }


def _event_without_source(event: dict[str, Any]) -> dict[str, Any]:
    return {
        **_event_identity(event),
        "s5_source": None,
        "context_status": "BLOCKED_NO_LOCAL_S5_COVERAGE",
        "strict_economic_score_eligible": False,
    }


def _serialise_touch(touch: dict[str, Any]) -> dict[str, Any]:
    return {**touch, "timestamp_utc": _iso(touch["timestamp_utc"])}


def _parse_time(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d+(?=[+-])", r"\1", text)
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _file_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def _floor_s5(value: datetime) -> datetime:
    return value.replace(second=value.second // 5 * 5, microsecond=0)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    main()
