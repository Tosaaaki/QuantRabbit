#!/usr/bin/env python3
"""Backfill and incrementally update trade pattern stats."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analysis.pattern_book import (  # noqa: E402
    PatternAction,
    PatternAggregate,
    build_pattern_id,
    classify_pattern_action,
)
from analysis.pattern_deep import DeepPatternConfig, run_pattern_deep_analysis  # noqa: E402

DEFAULT_TRADES_DB = BASE_DIR / "logs" / "trades.db"
DEFAULT_PATTERNS_DB = BASE_DIR / "logs" / "patterns.db"
DEFAULT_OUTPUT_JSON = BASE_DIR / "config" / "pattern_book.json"
DEFAULT_DEEP_OUTPUT_JSON = BASE_DIR / "config" / "pattern_book_deep.json"


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_iso_utc(value: Any) -> dt.datetime | None:
    if not value:
        return None
    raw = str(value)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS pattern_trade_features (
          transaction_id INTEGER PRIMARY KEY,
          ticket_id TEXT,
          close_time TEXT NOT NULL,
          open_time TEXT,
          hold_sec REAL NOT NULL,
          pocket TEXT NOT NULL,
          strategy_tag TEXT NOT NULL,
          direction TEXT NOT NULL,
          units INTEGER NOT NULL,
          pl_pips REAL NOT NULL,
          realized_pl REAL NOT NULL,
          pattern_id TEXT NOT NULL,
          signal_mode TEXT,
          mtf_gate TEXT,
          horizon_gate TEXT,
          extrema_reason TEXT,
          confidence INTEGER,
          spread_pips REAL,
          tp_pips REAL,
          sl_pips REAL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pattern_features_close_time
          ON pattern_trade_features(close_time);
        CREATE INDEX IF NOT EXISTS idx_pattern_features_pattern
          ON pattern_trade_features(pattern_id);

        CREATE TABLE IF NOT EXISTS pattern_stats (
          pattern_id TEXT PRIMARY KEY,
          strategy_tag TEXT NOT NULL,
          pocket TEXT NOT NULL,
          direction TEXT NOT NULL,
          trades INTEGER NOT NULL,
          wins INTEGER NOT NULL,
          losses INTEGER NOT NULL,
          win_rate REAL NOT NULL,
          avg_pips REAL NOT NULL,
          total_pips REAL NOT NULL,
          gross_profit REAL NOT NULL,
          gross_loss REAL NOT NULL,
          profit_factor REAL NOT NULL,
          avg_hold_sec REAL NOT NULL,
          last_close_time TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pattern_actions (
          pattern_id TEXT PRIMARY KEY,
          action TEXT NOT NULL,
          lot_multiplier REAL NOT NULL,
          reason TEXT NOT NULL,
          trades INTEGER NOT NULL,
          win_rate REAL NOT NULL,
          profit_factor REAL NOT NULL,
          avg_pips REAL NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )


def _extract_entry_thesis(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _last_transaction_id(patterns_con: sqlite3.Connection) -> int:
    row = patterns_con.execute(
        "SELECT COALESCE(MAX(transaction_id), 0) AS max_tx FROM pattern_trade_features"
    ).fetchone()
    if not row:
        return 0
    return _safe_int(row["max_tx"] if isinstance(row, sqlite3.Row) else row[0])


def _fetch_new_trades(
    trades_con: sqlite3.Connection,
    *,
    after_tx: int,
    batch_size: int,
) -> list[sqlite3.Row]:
    return trades_con.execute(
        """
        SELECT
          transaction_id,
          ticket_id,
          pocket,
          COALESCE(NULLIF(strategy_tag, ''), strategy, 'unknown') AS strategy_tag,
          units,
          pl_pips,
          realized_pl,
          open_time,
          close_time,
          entry_thesis
        FROM trades
        WHERE close_time IS NOT NULL
          AND transaction_id IS NOT NULL
          AND transaction_id > ?
        ORDER BY transaction_id ASC
        LIMIT ?
        """,
        (after_tx, batch_size),
    ).fetchall()


def _upsert_feature_rows(
    patterns_con: sqlite3.Connection,
    rows: list[sqlite3.Row],
    *,
    updated_at: str,
) -> int:
    payloads: list[tuple[Any, ...]] = []
    for row in rows:
        entry_thesis = _extract_entry_thesis(row["entry_thesis"])
        units = _safe_int(row["units"])
        strategy_tag = str(row["strategy_tag"] or "unknown")
        pocket = str(row["pocket"] or "unknown")
        pattern_id = build_pattern_id(
            entry_thesis=entry_thesis,
            units=units,
            pocket=pocket,
            strategy_tag_fallback=strategy_tag,
        )
        open_time = row["open_time"]
        close_time = row["close_time"]
        open_dt = _parse_iso_utc(open_time)
        close_dt = _parse_iso_utc(close_time)
        hold_sec = 0.0
        if open_dt and close_dt:
            hold_sec = max(0.0, (close_dt - open_dt).total_seconds())
        direction = "long" if units > 0 else "short" if units < 0 else "unknown"
        payloads.append(
            (
                _safe_int(row["transaction_id"]),
                str(row["ticket_id"] or ""),
                str(close_time or ""),
                str(open_time or ""),
                hold_sec,
                pocket,
                strategy_tag,
                direction,
                units,
                _safe_float(row["pl_pips"]),
                _safe_float(row["realized_pl"]),
                pattern_id,
                str(entry_thesis.get("signal_mode") or entry_thesis.get("entry_mode") or ""),
                str(entry_thesis.get("mtf_regime_gate") or entry_thesis.get("mtf_gate") or ""),
                str(entry_thesis.get("horizon_gate") or ""),
                str(entry_thesis.get("extrema_gate_reason") or ""),
                _safe_int(entry_thesis.get("confidence")),
                _safe_float(entry_thesis.get("spread_pips")),
                _safe_float(entry_thesis.get("tp_pips")),
                _safe_float(entry_thesis.get("sl_pips")),
                updated_at,
            )
        )

    if not payloads:
        return 0

    patterns_con.executemany(
        """
        INSERT INTO pattern_trade_features (
          transaction_id,
          ticket_id,
          close_time,
          open_time,
          hold_sec,
          pocket,
          strategy_tag,
          direction,
          units,
          pl_pips,
          realized_pl,
          pattern_id,
          signal_mode,
          mtf_gate,
          horizon_gate,
          extrema_reason,
          confidence,
          spread_pips,
          tp_pips,
          sl_pips,
          updated_at
        ) VALUES (
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(transaction_id) DO UPDATE SET
          ticket_id=excluded.ticket_id,
          close_time=excluded.close_time,
          open_time=excluded.open_time,
          hold_sec=excluded.hold_sec,
          pocket=excluded.pocket,
          strategy_tag=excluded.strategy_tag,
          direction=excluded.direction,
          units=excluded.units,
          pl_pips=excluded.pl_pips,
          realized_pl=excluded.realized_pl,
          pattern_id=excluded.pattern_id,
          signal_mode=excluded.signal_mode,
          mtf_gate=excluded.mtf_gate,
          horizon_gate=excluded.horizon_gate,
          extrema_reason=excluded.extrema_reason,
          confidence=excluded.confidence,
          spread_pips=excluded.spread_pips,
          tp_pips=excluded.tp_pips,
          sl_pips=excluded.sl_pips,
          updated_at=excluded.updated_at
        """,
        payloads,
    )
    return len(payloads)


def _aggregate(
    patterns_con: sqlite3.Connection,
    *,
    cutoff_iso: str,
) -> list[sqlite3.Row]:
    return patterns_con.execute(
        """
        SELECT
          pattern_id,
          MIN(strategy_tag) AS strategy_tag,
          MIN(pocket) AS pocket,
          MIN(direction) AS direction,
          COUNT(*) AS trades,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
          SUM(pl_pips) AS total_pips,
          AVG(pl_pips) AS avg_pips,
          SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS gross_profit,
          SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS gross_loss,
          AVG(hold_sec) AS avg_hold_sec,
          MAX(close_time) AS last_close_time
        FROM pattern_trade_features
        WHERE close_time >= ?
        GROUP BY pattern_id
        ORDER BY trades DESC, pattern_id
        """,
        (cutoff_iso,),
    ).fetchall()


def _to_actionable_row(
    row: sqlite3.Row,
    *,
    min_samples_soft: int,
    min_samples_block: int,
) -> tuple[dict[str, Any], PatternAction]:
    trades = _safe_int(row["trades"])
    wins = _safe_int(row["wins"])
    losses = _safe_int(row["losses"])
    total_pips = _safe_float(row["total_pips"])
    avg_pips = _safe_float(row["avg_pips"])
    gross_profit = _safe_float(row["gross_profit"])
    gross_loss = _safe_float(row["gross_loss"])
    win_rate = (wins / trades) if trades > 0 else 0.0
    if gross_loss <= 0.0:
        profit_factor = 999.0 if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss
    aggregate = PatternAggregate(
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_pips=avg_pips,
        total_pips=total_pips,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
    )
    action = classify_pattern_action(
        aggregate,
        min_samples_soft=min_samples_soft,
        min_samples_block=min_samples_block,
    )
    stats_row = {
        "pattern_id": str(row["pattern_id"]),
        "strategy_tag": str(row["strategy_tag"] or "unknown"),
        "pocket": str(row["pocket"] or "unknown"),
        "direction": str(row["direction"] or "unknown"),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "avg_pips": round(avg_pips, 4),
        "total_pips": round(total_pips, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_hold_sec": round(_safe_float(row["avg_hold_sec"]), 2),
        "last_close_time": str(row["last_close_time"] or ""),
        "action": action.action,
        "lot_multiplier": round(action.lot_multiplier, 3),
        "reason": action.reason,
    }
    return stats_row, action


def _write_aggregates(
    patterns_con: sqlite3.Connection,
    *,
    rows: list[dict[str, Any]],
    as_of: str,
) -> None:
    patterns_con.execute("DELETE FROM pattern_stats")
    patterns_con.execute("DELETE FROM pattern_actions")
    if not rows:
        return
    patterns_con.executemany(
        """
        INSERT INTO pattern_stats (
          pattern_id, strategy_tag, pocket, direction, trades, wins, losses,
          win_rate, avg_pips, total_pips, gross_profit, gross_loss, profit_factor,
          avg_hold_sec, last_close_time, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["pattern_id"],
                r["strategy_tag"],
                r["pocket"],
                r["direction"],
                r["trades"],
                r["wins"],
                r["losses"],
                r["win_rate"],
                r["avg_pips"],
                r["total_pips"],
                r["gross_profit"],
                r["gross_loss"],
                r["profit_factor"],
                r["avg_hold_sec"],
                r["last_close_time"],
                as_of,
            )
            for r in rows
        ],
    )
    patterns_con.executemany(
        """
        INSERT INTO pattern_actions (
          pattern_id, action, lot_multiplier, reason, trades, win_rate,
          profit_factor, avg_pips, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["pattern_id"],
                r["action"],
                r["lot_multiplier"],
                r["reason"],
                r["trades"],
                r["win_rate"],
                r["profit_factor"],
                r["avg_pips"],
                as_of,
            )
            for r in rows
        ],
    )


def _write_json(
    *,
    output_path: Path,
    as_of: str,
    lookback_days: int,
    processed_new_rows: int,
    rows: list[dict[str, Any]],
    min_samples_soft: int,
    min_samples_block: int,
    deep_summary: dict[str, Any] | None = None,
) -> None:
    action_counts: dict[str, int] = {}
    for row in rows:
        action = row["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    top_edges = sorted(
        [r for r in rows if r["action"] == "boost"],
        key=lambda r: (r["lot_multiplier"], r["avg_pips"], r["trades"]),
        reverse=True,
    )[:25]
    weak_edges = sorted(
        [r for r in rows if r["action"] in {"reduce", "block"}],
        key=lambda r: (r["lot_multiplier"], r["avg_pips"], -r["trades"]),
    )[:25]
    payload = {
        "as_of": as_of,
        "lookback_days": lookback_days,
        "processed_new_rows": processed_new_rows,
        "patterns_total": len(rows),
        "min_samples_soft": min_samples_soft,
        "min_samples_block": min_samples_block,
        "action_counts": action_counts,
        "top_edges": top_edges,
        "weak_edges": weak_edges,
        "deep_analysis": deep_summary or {},
        "patterns": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def run(args: argparse.Namespace) -> dict[str, Any]:
    now_utc = dt.datetime.now(dt.timezone.utc)
    as_of = now_utc.isoformat(timespec="seconds")
    cutoff_iso = (now_utc - dt.timedelta(days=max(1, args.lookback_days))).isoformat()

    trades_db_path = Path(args.trades_db).resolve()
    patterns_db_path = Path(args.patterns_db).resolve()
    patterns_db_path.parent.mkdir(parents=True, exist_ok=True)
    deep_output_path = Path(args.deep_output_json).resolve()
    deep_summary: dict[str, Any] = {}

    with sqlite3.connect(trades_db_path, timeout=30.0, isolation_level=None) as trades_con:
        trades_con.row_factory = sqlite3.Row
        trades_con.execute("PRAGMA query_only=ON")
        with sqlite3.connect(patterns_db_path, timeout=30.0) as patterns_con:
            patterns_con.row_factory = sqlite3.Row
            _ensure_schema(patterns_con)
            last_tx = _last_transaction_id(patterns_con)

            processed_rows = 0
            cursor_tx = last_tx
            while processed_rows < args.max_backfill_rows:
                remaining = max(1, args.max_backfill_rows - processed_rows)
                batch_size = min(args.batch_size, remaining)
                rows = _fetch_new_trades(
                    trades_con,
                    after_tx=cursor_tx,
                    batch_size=batch_size,
                )
                if not rows:
                    break
                processed = _upsert_feature_rows(
                    patterns_con,
                    rows,
                    updated_at=as_of,
                )
                patterns_con.commit()
                processed_rows += processed
                cursor_tx = _safe_int(rows[-1]["transaction_id"])
                if len(rows) < batch_size:
                    break

            agg_rows = _aggregate(patterns_con, cutoff_iso=cutoff_iso)
            stat_rows: list[dict[str, Any]] = []
            for row in agg_rows:
                parsed, _action = _to_actionable_row(
                    row,
                    min_samples_soft=max(1, args.min_samples_soft),
                    min_samples_block=max(1, args.min_samples_block),
                )
                stat_rows.append(parsed)
            _write_aggregates(patterns_con, rows=stat_rows, as_of=as_of)
            deep_summary = run_pattern_deep_analysis(
                patterns_con,
                cutoff_iso=cutoff_iso,
                as_of=as_of,
                output_path=deep_output_path,
                config=DeepPatternConfig(
                    min_samples=max(1, args.deep_min_samples),
                    prior_strength=max(1, args.deep_prior_strength),
                    recent_days=max(1, args.deep_recent_days),
                    baseline_days=max(1, args.deep_baseline_days),
                    min_recent_samples=max(1, args.deep_min_recent_samples),
                    min_prev_samples=max(1, args.deep_min_prev_samples),
                    bootstrap_samples=max(80, args.deep_bootstrap_samples),
                    cluster_min=max(2, args.deep_cluster_min),
                    cluster_max=max(2, args.deep_cluster_max),
                    cluster_min_samples=max(5, args.deep_cluster_min_samples),
                    random_state=max(1, args.deep_random_state),
                ),
            )
            patterns_con.commit()

    _write_json(
        output_path=Path(args.output_json).resolve(),
        as_of=as_of,
        lookback_days=max(1, args.lookback_days),
        processed_new_rows=processed_rows,
        rows=stat_rows,
        min_samples_soft=max(1, args.min_samples_soft),
        min_samples_block=max(1, args.min_samples_block),
        deep_summary=deep_summary,
    )
    return {
        "as_of": as_of,
        "processed_new_rows": processed_rows,
        "patterns_total": len(stat_rows),
        "deep_patterns_scored": deep_summary.get("patterns_scored", 0),
        "deep_drift_rows": deep_summary.get("drift_rows", 0),
        "deep_cluster_count": deep_summary.get("cluster_count", 0),
        "output_json": str(Path(args.output_json).resolve()),
        "deep_output_json": str(deep_output_path),
        "patterns_db": str(patterns_db_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill/update pattern stats from trades.db")
    parser.add_argument("--trades-db", type=Path, default=DEFAULT_TRADES_DB)
    parser.add_argument("--patterns-db", type=Path, default=DEFAULT_PATTERNS_DB)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--deep-output-json", type=Path, default=DEFAULT_DEEP_OUTPUT_JSON)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--max-backfill-rows", type=int, default=500000)
    parser.add_argument("--min-samples-soft", type=int, default=30)
    parser.add_argument("--min-samples-block", type=int, default=120)
    parser.add_argument("--deep-min-samples", type=int, default=30)
    parser.add_argument("--deep-prior-strength", type=int, default=24)
    parser.add_argument("--deep-recent-days", type=int, default=5)
    parser.add_argument("--deep-baseline-days", type=int, default=30)
    parser.add_argument("--deep-min-recent-samples", type=int, default=8)
    parser.add_argument("--deep-min-prev-samples", type=int, default=20)
    parser.add_argument("--deep-bootstrap-samples", type=int, default=240)
    parser.add_argument("--deep-cluster-min", type=int, default=3)
    parser.add_argument("--deep-cluster-max", type=int, default=8)
    parser.add_argument("--deep-cluster-min-samples", type=int, default=20)
    parser.add_argument("--deep-random-state", type=int, default=42)
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
