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
from workers.common.setup_context import extract_setup_identity


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


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


def _extract_entry_thesis(payload: dict[str, Any]) -> dict[str, Any]:
    thesis = payload.get("entry_thesis")
    return thesis if isinstance(thesis, dict) else {}


def _extract_setup_context(payload: dict[str, Any]) -> dict[str, str]:
    thesis = _extract_entry_thesis(payload)
    if not thesis:
        return {}
    units = _safe_int(payload.get("units"), 0)
    if units == 0:
        units = _safe_int(thesis.get("entry_units_intent"), 0)
    if units == 0:
        oanda = payload.get("oanda")
        if isinstance(oanda, dict):
            order = oanda.get("order")
            if isinstance(order, dict):
                units = _safe_int(order.get("units"), 0)
    return extract_setup_identity(thesis, units=units)


def _setup_bucket_key(context: dict[str, str]) -> str:
    setup_fingerprint = str(context.get("setup_fingerprint") or "").strip()
    if setup_fingerprint:
        return setup_fingerprint
    flow_regime = str(context.get("flow_regime") or "").strip()
    microstructure_bucket = str(context.get("microstructure_bucket") or "").strip()
    if flow_regime and microstructure_bucket:
        return f"{flow_regime}|{microstructure_bucket}"
    if flow_regime:
        return flow_regime
    return microstructure_bucket


def _setup_match_dimension(context: dict[str, str]) -> str:
    if str(context.get("setup_fingerprint") or "").strip():
        return "setup_fingerprint"
    if str(context.get("flow_regime") or "").strip() and str(context.get("microstructure_bucket") or "").strip():
        return "flow_micro"
    if str(context.get("flow_regime") or "").strip():
        return "flow_regime"
    if str(context.get("microstructure_bucket") or "").strip():
        return "microstructure_bucket"
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
                "  AND julianday(ts) >= julianday('now', ?) "
                "ORDER BY julianday(ts) DESC, ts DESC"
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
                "setups": {},
            },
        )
        setup_context = _extract_setup_context(payload)
        setup_bucket = None
        setup_key = _setup_bucket_key(setup_context) if setup_context else ""
        if setup_key:
            setup_bucket = bucket["setups"].setdefault(
                setup_key,
                {
                    "setup_key": setup_key,
                    "match_dimension": _setup_match_dimension(setup_context),
                    "setup_fingerprint": str(setup_context.get("setup_fingerprint") or ""),
                    "flow_regime": str(setup_context.get("flow_regime") or ""),
                    "microstructure_bucket": str(setup_context.get("microstructure_bucket") or ""),
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
        if isinstance(setup_bucket, dict):
            setup_bucket["status_counts"][status] += 1
        if status == "preflight_start":
            bucket["preflights"] += 1
            if isinstance(setup_bucket, dict):
                setup_bucket["preflights"] += 1
        elif status == "submit_attempt":
            bucket["submit_attempts"] += 1
            if isinstance(setup_bucket, dict):
                setup_bucket["submit_attempts"] += 1
        elif status == "filled":
            bucket["filled"] += 1
            if isinstance(setup_bucket, dict):
                setup_bucket["filled"] += 1
        elif status == "probability_scaled":
            bucket["soft_reduces"] += 1
            if isinstance(setup_bucket, dict):
                setup_bucket["soft_reduces"] += 1
        elif _is_hard_block_status(status):
            bucket["hard_blocks"] += 1
            bucket["blockers"][f"status:{status}"] += 1
            if isinstance(setup_bucket, dict):
                setup_bucket["hard_blocks"] += 1
                setup_bucket["blockers"][f"status:{status}"] += 1

        for trail_item in _extract_trail(payload):
            stage = str(trail_item.get("stage") or "").strip() or "unknown"
            stage_status = str(trail_item.get("status") or "").strip() or "unknown"
            reason = str(trail_item.get("reason") or "").strip()
            stage_key = f"{stage}:{stage_status}"
            global_stage_counts[stage_key] += 1
            bucket["stage_counts"][stage_key] += 1
            if isinstance(setup_bucket, dict):
                setup_bucket["stage_counts"][stage_key] += 1
            if stage_status == "reduce":
                bucket["soft_reduces"] += 1
                if isinstance(setup_bucket, dict):
                    setup_bucket["soft_reduces"] += 1
            if stage_status == "block":
                bucket["hard_blocks"] += 1
                blocker_key = f"{stage}:{reason or 'unknown'}"
                bucket["blockers"][blocker_key] += 1
                if isinstance(setup_bucket, dict):
                    setup_bucket["hard_blocks"] += 1
                    setup_bucket["blockers"][blocker_key] += 1

    total_attempts = sum(int(bucket["preflights"]) for bucket in strategies.values())
    total_fills = sum(int(bucket["filled"]) for bucket in strategies.values())
    strategy_rows: dict[str, Any] = {}
    for key, bucket in strategies.items():
        preflights = int(bucket["preflights"])
        filled = int(bucket["filled"])
        hard_blocks = int(bucket["hard_blocks"])
        attempt_share = preflights / float(max(1, total_attempts))
        fill_share = filled / float(max(1, total_fills))
        total_setup_attempts = sum(
            int(item.get("preflights") or 0) for item in bucket.get("setups", {}).values()
        )
        total_setup_fills = sum(
            int(item.get("filled") or 0) for item in bucket.get("setups", {}).values()
        )
        status_counts_map = {
            str(name): int(count)
            for name, count in sorted(bucket["status_counts"].items(), key=lambda item: item[0])
        }
        setup_rows: list[dict[str, Any]] = []
        for setup_key, setup_bucket in sorted(
            bucket.get("setups", {}).items(),
            key=lambda item: (-int(item[1].get("preflights") or 0), item[0]),
        ):
            setup_preflights = int(setup_bucket["preflights"])
            setup_filled = int(setup_bucket["filled"])
            setup_hard_blocks = int(setup_bucket["hard_blocks"])
            if setup_preflights <= 0 and setup_filled <= 0 and setup_hard_blocks <= 0:
                continue
            setup_attempt_share = setup_preflights / float(max(1, total_setup_attempts))
            setup_fill_share = setup_filled / float(max(1, total_setup_fills))
            setup_status_counts = {
                str(name): int(count)
                for name, count in sorted(setup_bucket["status_counts"].items(), key=lambda item: item[0])
            }
            setup_rows.append(
                {
                    "setup_key": setup_key,
                    "match_dimension": str(setup_bucket.get("match_dimension") or "unknown"),
                    "setup_fingerprint": str(setup_bucket.get("setup_fingerprint") or ""),
                    "flow_regime": str(setup_bucket.get("flow_regime") or ""),
                    "microstructure_bucket": str(setup_bucket.get("microstructure_bucket") or ""),
                    "attempts": setup_preflights,
                    "preflights": setup_preflights,
                    "fills": setup_filled,
                    "submit_attempts": int(setup_bucket["submit_attempts"]),
                    "filled": setup_filled,
                    "hard_blocks": setup_hard_blocks,
                    "soft_reduces": int(setup_bucket["soft_reduces"]),
                    "filled_rate": round(setup_filled / float(max(1, setup_preflights)), 4),
                    "fill_rate": round(setup_filled / float(max(1, setup_preflights)), 4),
                    "attempt_share": round(setup_attempt_share, 6),
                    "fill_share": round(setup_fill_share, 6),
                    "share_gap": round(setup_attempt_share - setup_fill_share, 6),
                    "hard_block_rate": round(setup_hard_blocks / float(max(1, setup_preflights)), 4),
                    "terminal_status_counts": setup_status_counts,
                    "status_counts": _sort_counts(setup_bucket["status_counts"]),
                    "stage_counts": _sort_counts(setup_bucket["stage_counts"]),
                    "top_blockers": _sort_counts(setup_bucket["blockers"], top_k=max(1, int(top_k))),
                }
            )
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
            "setups": setup_rows,
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
