#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.strategy_tags import extract_strategy_tags


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _is_entry_status(status: str) -> bool:
    if not status:
        return False
    return not status.startswith("close_")


def _is_hard_block_status(status: str) -> bool:
    status_l = status.lower()
    if status_l in {"preflight_start", "submit_attempt", "filled", "brain_shadow", "probability_scaled"}:
        return False
    return any(token in status_l for token in ("block", "reject", "disabled", "cooldown", "rejected"))


def _extract_strategy_tag(payload: dict[str, Any]) -> tuple[str, str]:
    thesis = payload.get("entry_thesis") if isinstance(payload.get("entry_thesis"), dict) else {}
    raw_tag, canonical_tag = extract_strategy_tags(
        strategy_tag=payload.get("strategy_tag"),
        strategy=payload.get("strategy"),
        entry_thesis=thesis,
    )
    strategy_key = raw_tag or canonical_tag or "unknown"
    strategy_canonical = canonical_tag or strategy_key
    return strategy_key, strategy_canonical


def _extract_pocket(payload: dict[str, Any], row_pocket: Any) -> str:
    thesis = payload.get("entry_thesis") if isinstance(payload.get("entry_thesis"), dict) else {}
    for raw in (row_pocket, payload.get("pocket"), thesis.get("pocket")):
        text = str(raw or "").strip().lower()
        if text:
            return text
    return "unknown"


def _extract_trail(payload: dict[str, Any]) -> list[dict[str, Any]]:
    thesis = payload.get("entry_thesis") if isinstance(payload.get("entry_thesis"), dict) else {}
    for container in (payload, thesis):
        trail = container.get("entry_path_attribution")
        if isinstance(trail, list):
            return [item for item in trail if isinstance(item, dict)]
    return []


def _sort_counts(counts: dict[str, int], *, top_k: int | None = None) -> list[dict[str, Any]]:
    rows = [
        {"key": key, "count": int(count)}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    if top_k is not None:
        return rows[:top_k]
    return rows


def build_report(db_path: Path, *, lookback_hours: int, limit: int, top_k: int) -> dict[str, Any]:
    rows: list[tuple[str, str, str]] = []
    if db_path.exists():
        uri = f"file:{db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=10.0, isolation_level=None) as conn:
            cur = conn.cursor()
            sql = (
                "SELECT ts, pocket, status, request_json "
                "FROM orders "
                "WHERE request_json IS NOT NULL "
                "  AND LENGTH(request_json) > 2 "
                "  AND ts >= datetime('now', ?) "
                "ORDER BY ts DESC"
            )
            params: list[Any] = [f"-{int(lookback_hours)} hours"]
            if int(limit) > 0:
                sql += " LIMIT ?"
                params.append(int(limit))
            cur.execute(sql, params)
            rows = [(str(ts or ""), str(pocket or ""), str(status or ""), request_json) for ts, pocket, status, request_json in cur.fetchall()]

    global_status_counts: dict[str, int] = defaultdict(int)
    global_stage_counts: dict[str, int] = defaultdict(int)
    strategies: dict[str, dict[str, Any]] = {}
    orders_considered = 0

    for _ts, pocket_raw, status_raw, request_json in rows:
        status = str(status_raw or "").strip()
        if not _is_entry_status(status):
            continue
        orders_considered += 1
        payload = _safe_json_loads(request_json)
        strategy, strategy_canonical = _extract_strategy_tag(payload)
        pocket = _extract_pocket(payload, pocket_raw)
        strategy_key = strategy
        if strategy_key in strategies and strategies[strategy_key].get("pocket") not in {pocket, None}:
            strategy_key = f"{strategy}:{pocket}"
        bucket = strategies.setdefault(
            strategy_key,
            {
                "strategy_key": strategy,
                "strategy_canonical": strategy_canonical,
                "pocket": pocket,
                "preflights": 0,
                "submit_attempts": 0,
                "filled": 0,
                "hard_blocks": 0,
                "soft_reduces": 0,
                "status_counts": defaultdict(int),
                "stage_counts": defaultdict(int),
                "blockers": defaultdict(int),
            },
        )
        global_status_counts[status] += 1
        bucket["status_counts"][status] += 1
        if status == "preflight_start":
            bucket["preflights"] += 1
        elif status == "submit_attempt":
            bucket["submit_attempts"] += 1
        elif status == "filled":
            bucket["filled"] += 1
        elif status == "probability_scaled":
            bucket["soft_reduces"] += 1
        elif _is_hard_block_status(status):
            bucket["hard_blocks"] += 1
            bucket["blockers"][f"status:{status}"] += 1

        for trail_item in _extract_trail(payload):
            stage = str(trail_item.get("stage") or "").strip() or "unknown"
            stage_status = str(trail_item.get("status") or "").strip() or "unknown"
            reason = str(trail_item.get("reason") or "").strip()
            stage_key = f"{stage}:{stage_status}"
            global_stage_counts[stage_key] += 1
            bucket["stage_counts"][stage_key] += 1
            if stage_status == "reduce":
                bucket["soft_reduces"] += 1
            if stage_status == "block":
                bucket["hard_blocks"] += 1
                blocker_key = f"{stage}:{reason or 'unknown'}"
                bucket["blockers"][blocker_key] += 1

    total_attempts = sum(int(bucket["preflights"]) for bucket in strategies.values())
    total_fills = sum(int(bucket["filled"]) for bucket in strategies.values())
    strategy_rows: dict[str, Any] = {}
    for key, bucket in strategies.items():
        preflights = int(bucket["preflights"])
        filled = int(bucket["filled"])
        hard_blocks = int(bucket["hard_blocks"])
        attempt_share = preflights / float(max(1, total_attempts))
        fill_share = filled / float(max(1, total_fills))
        status_counts_map = {
            str(name): int(count)
            for name, count in sorted(bucket["status_counts"].items(), key=lambda item: item[0])
        }
        strategy_rows[key] = {
            "strategy_key": bucket["strategy_key"],
            "strategy_canonical": bucket["strategy_canonical"],
            "pocket": bucket["pocket"],
            "attempts": preflights,
            "preflights": preflights,
            "fills": filled,
            "submit_attempts": int(bucket["submit_attempts"]),
            "filled": filled,
            "hard_blocks": hard_blocks,
            "soft_reduces": int(bucket["soft_reduces"]),
            "filled_rate": round(filled / float(max(1, preflights)), 4),
            "fill_rate": round(filled / float(max(1, preflights)), 4),
            "attempt_share": round(attempt_share, 6),
            "fill_share": round(fill_share, 6),
            "share_gap": round(attempt_share - fill_share, 6),
            "hard_block_rate": round(hard_blocks / float(max(1, preflights)), 4),
            "terminal_status_counts": status_counts_map,
            "status_counts": _sort_counts(bucket["status_counts"]),
            "stage_counts": _sort_counts(bucket["stage_counts"]),
            "top_blockers": _sort_counts(bucket["blockers"], top_k=max(1, int(top_k))),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_hours": int(lookback_hours),
        "rows": len(rows),
        "orders_considered": int(orders_considered),
        "strategies_count": len(strategy_rows),
        "status_counts": _sort_counts(global_status_counts),
        "stage_counts": _sort_counts(global_stage_counts),
        "strategies": strategy_rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate entry-path attribution from orders.db")
    ap.add_argument("--orders-db", default="logs/orders.db")
    ap.add_argument("--output", default="logs/entry_path_summary_latest.json")
    ap.add_argument("--history", default="logs/entry_path_summary_history.jsonl")
    ap.add_argument("--lookback-hours", type=int, default=24)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--top-k", type=int, default=6)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_report(
        Path(args.orders_db).resolve(),
        lookback_hours=max(1, int(args.lookback_hours)),
        limit=max(0, int(args.limit)),
        top_k=max(1, int(args.top_k)),
    )
    _write_json_atomic(Path(args.output).resolve(), payload)
    _append_jsonl(Path(args.history).resolve(), payload)
    print(
        f"[entry-path-aggregator] wrote {args.output} "
        f"strategies={payload['strategies_count']} rows={payload['rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
