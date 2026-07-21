#!/usr/bin/env python3
"""Append one evidence-bound DOJO profitability observation.

The forward ledger remains untouched.  This writer validates that ledger and
its resumable checkpoint, derives cohort economics, and appends a separate
hash-chained research record only when settled/open-trade evidence changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
ZERO_SHA = "0" * 64
SETTLED_EVENTS = {"EXIT_TP", "EXIT_SL", "CLOSE", "MARGIN_CLOSEOUT"}
DECISIONS = {"KEEP", "TEST", "REJECT"}


def _sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_ledger(path: Path) -> tuple[list[dict[str, Any]], str]:
    events: list[dict[str, Any]] = []
    previous = ZERO_SHA
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                body = {
                    key: record[key]
                    for key in ("ts_utc", "event", "payload", "prev_sha")
                }
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid ledger row {line_number}") from exc
            if record["prev_sha"] != previous:
                raise ValueError(f"ledger prev_sha mismatch at row {line_number}")
            if record.get("sha") != _sha(body):
                raise ValueError(f"ledger sha mismatch at row {line_number}")
            previous = record["sha"]
            events.append(record)
    if not events:
        raise ValueError("forward ledger is empty")
    return events, previous


def _trade_number(trade_id: str | None) -> int | None:
    if not trade_id or len(trade_id) < 2 or not trade_id[1:].isdigit():
        return None
    return int(trade_id[1:])


def _metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(item["pl_jpy"]) for item in outcomes]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = gross_win / len(wins) if wins else None
    avg_loss = gross_loss / len(losses) if losses else None
    payoff = avg_win / avg_loss if avg_win is not None and avg_loss else None
    breakeven = 1 / (1 + payoff) if payoff is not None else None
    return {
        "settled": len(values),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(values) if values else None,
        "net_jpy": round(sum(values), 2),
        "profit_factor": gross_win / gross_loss if gross_loss else None,
        "expectancy_jpy": sum(values) / len(values) if values else None,
        "average_win_jpy": avg_win,
        "average_loss_jpy": avg_loss,
        "payoff_ratio": payoff,
        "breakeven_win_rate": breakeven,
        "worst_loss_jpy": min(losses) if losses else None,
    }


def _load_knowledge(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        return [], ZERO_SHA
    rows: list[dict[str, Any]] = []
    previous = ZERO_SHA
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                body = {key: row[key] for key in row if key != "sha"}
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid knowledge row {line_number}") from exc
            if row.get("prev_sha") != previous:
                raise ValueError(f"knowledge prev_sha mismatch at row {line_number}")
            if row.get("sha") != _sha(body):
                raise ValueError(f"knowledge sha mismatch at row {line_number}")
            previous = row["sha"]
            rows.append(row)
    return rows, previous


def build_observation(
    session_dir: Path,
    *,
    clean_after: int,
    carry_ids: set[str],
    hypothesis: str,
    counterevidence: str,
    decision: str,
    next_test: str,
) -> dict[str, Any]:
    events, terminal_sha = _read_ledger(session_dir / "ledger.jsonl")
    state = json.loads((session_dir / "state.json").read_text())
    snapshot = json.loads((session_dir / "broker_snapshot.json").read_text())
    if snapshot.get("ledger_sha") != terminal_sha:
        raise ValueError("snapshot ledger_sha does not match terminal ledger sha")

    fills: dict[str, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") or {}
        trade_id = payload.get("trade_id")
        if event["event"].startswith("FILL_") and trade_id:
            fills[trade_id] = event
        if event["event"] in SETTLED_EVENTS and payload.get("pl_jpy") is not None:
            fill_payload = (fills.get(trade_id) or {}).get("payload") or {}
            outcomes.append(
                {
                    "trade_id": trade_id,
                    "event": event["event"],
                    "ts_utc": event["ts_utc"],
                    "pl_jpy": float(payload["pl_jpy"]),
                    "strategy_tag": payload.get("strategy_tag")
                    or fill_payload.get("strategy_tag"),
                }
            )

    clean = [
        item
        for item in outcomes
        if (_trade_number(item["trade_id"]) or -1) > clean_after
    ]
    carry = [item for item in outcomes if item["trade_id"] in carry_ids]
    wall_time = datetime.fromisoformat(state["wall_time_utc"])
    jst_day = wall_time.astimezone(JST).date()
    daily = [
        item
        for item in outcomes
        if datetime.fromisoformat(item["ts_utc"]).astimezone(JST).date() == jst_day
    ]
    account = state.get("account") or {}
    balance = float(account.get("balance_jpy") or 0.0)
    daily_net = sum(float(item["pl_jpy"]) for item in daily)
    day_open = balance - daily_net
    target = day_open * 0.02
    open_positions = [
        {
            "trade_id": pos.get("trade_id"),
            "strategy_tag": pos.get("strategy_tag"),
            "pair": pos.get("pair"),
            "side": pos.get("side"),
            "units": pos.get("units"),
            "opened_ts": pos.get("opened_ts"),
        }
        for pos in state.get("positions") or []
    ]
    evidence_id = _sha(
        {
            "ledger_terminal_sha": terminal_sha,
            "open_positions": open_positions,
        }
    )
    return {
        "contract": "QR_DOJO_PROFITABILITY_KNOWLEDGE_V1",
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "evidence_id": evidence_id,
        "ledger_terminal_sha": terminal_sha,
        "state_wall_time_utc": state["wall_time_utc"],
        "cohorts": {
            "all": _metrics(outcomes),
            "carry_repair": _metrics(carry),
            "clean_postfix": _metrics(clean),
            "jst_day": {"date": jst_day.isoformat(), **_metrics(daily)},
        },
        "account": {
            "balance_jpy": balance,
            "equity_jpy": account.get("equity_jpy"),
            "margin_usage": account.get("margin_usage"),
            "open_positions": open_positions,
        },
        "daily_2pct_benchmark": {
            "day_opening_balance_jpy": round(day_open, 2),
            "target_jpy": round(target, 2),
            "achieved_jpy": round(daily_net, 2),
            "achieved_return": daily_net / day_open if day_open > 0 else None,
            "excess_or_shortfall_jpy": round(daily_net - target, 2),
            "guarantee": False,
        },
        "analysis": {
            "hypothesis": hypothesis,
            "counterevidence": counterevidence,
            "decision": decision,
            "next_falsifiable_test": next_test,
        },
    }


def _bounded(value: str, field: str) -> str:
    value = value.strip()
    if not value or len(value) > 600:
        raise ValueError(f"{field} must contain 1..600 characters")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--knowledge", type=Path, required=True)
    parser.add_argument("--clean-after", type=int, default=229)
    parser.add_argument("--carry-id", action="append", default=[])
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--counterevidence", required=True)
    parser.add_argument("--decision", choices=sorted(DECISIONS), required=True)
    parser.add_argument("--next-test", required=True)
    args = parser.parse_args()
    observation = build_observation(
        args.session_dir,
        clean_after=args.clean_after,
        carry_ids=set(args.carry_id),
        hypothesis=_bounded(args.hypothesis, "hypothesis"),
        counterevidence=_bounded(args.counterevidence, "counterevidence"),
        decision=args.decision,
        next_test=_bounded(args.next_test, "next_test"),
    )
    rows, previous = _load_knowledge(args.knowledge)
    if rows and rows[-1].get("evidence_id") == observation["evidence_id"]:
        print(json.dumps({"status": "NO_CHANGE", "evidence_id": observation["evidence_id"]}))
        return 0
    observation["prev_sha"] = previous
    observation["sha"] = _sha(observation)
    args.knowledge.parent.mkdir(parents=True, exist_ok=True)
    with args.knowledge.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(observation, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "status": "APPENDED",
        "evidence_id": observation["evidence_id"],
        "sha": observation["sha"],
        "cohorts": observation["cohorts"],
        "daily_2pct_benchmark": observation["daily_2pct_benchmark"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
