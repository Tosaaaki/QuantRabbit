#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill trades.entry_thesis contract fields from logs/orders.db.

Goal:
- Ensure `entry_probability` and `entry_units_intent` exist inside `trades.entry_thesis`
  for non-manual trades so pattern learning / RCA queries have a consistent payload.

Primary source:
- orders.status='submit_attempt' -> request_json -> entry_thesis (or meta.entry_thesis)

Fallback (when orders lookup is unavailable):
- keep existing entry_thesis keys and add:
  - entry_probability = 1.0
  - entry_units_intent = abs(trades.units)

Safety:
- Takes an exclusive file lock on logs/trades.db.lock (same lock path used by position_manager).
- Creates a timestamped SQLite backup before writing (unless --no-backup).

Usage:
  python scripts/backfill_entry_thesis_from_orders.py --dry-run
  python scripts/backfill_entry_thesis_from_orders.py
  python scripts/backfill_entry_thesis_from_orders.py --until-utc 2026-02-27T23:59:59+00:00
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import sqlite3
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill trades.entry_thesis from orders.db (contract fields)")
    p.add_argument("--trades-db", default="logs/trades.db", help="Path to trades.db")
    p.add_argument("--orders-db", default="logs/orders.db", help="Path to orders.db")
    p.add_argument(
        "--until-utc",
        default=None,
        help="Only backfill trades with close_time <= this ISO8601 UTC timestamp (e.g. 2026-02-27T23:59:59+00:00)",
    )
    p.add_argument("--limit", type=int, default=0, help="Max rows to scan (0 = no limit)")
    p.add_argument("--dry-run", action="store_true", help="Do not write anything")
    p.add_argument("--no-backup", action="store_true", help="Skip pre-write backup (not recommended)")
    p.add_argument(
        "--backup-dir",
        default="logs/archive",
        help="Directory to store backup copy of trades.db",
    )
    p.add_argument("--print-samples", type=int, default=8, help="Print N sample updates")
    return p.parse_args()


def _parse_iso8601_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _safe_json_obj(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return dict(loaded)
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_entry_probability(value: Any) -> float | None:
    prob = _coerce_float(value)
    if prob is None:
        return None
    if not (0.0 <= prob <= 1.0):
        return None
    return float(prob)


def _normalize_units_intent(value: Any) -> int | None:
    units = _coerce_float(value)
    if units is None:
        return None
    try:
        as_int = int(abs(units))
    except (TypeError, ValueError):
        return None
    if as_int < 0:
        return None
    return as_int


def _extract_thesis_from_request(request_json: Any) -> dict[str, Any] | None:
    payload = _safe_json_obj(request_json)
    if payload is None:
        return None
    thesis: dict[str, Any] = {}

    raw_thesis = payload.get("entry_thesis")
    if isinstance(raw_thesis, dict):
        thesis.update(raw_thesis)
    else:
        meta = payload.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("entry_thesis"), dict):
            thesis.update(meta["entry_thesis"])

    # Ensure contract fields are captured even if logged only at top level.
    if thesis.get("entry_probability") is None:
        prob = _normalize_entry_probability(payload.get("entry_probability"))
        if prob is not None:
            thesis["entry_probability"] = prob
    if thesis.get("entry_units_intent") is None:
        units = _normalize_units_intent(
            payload.get("entry_units_intent")
            or payload.get("entry_units")
            or payload.get("units")
        )
        if units is not None:
            thesis["entry_units_intent"] = units

    return thesis or None


@contextlib.contextmanager
def _file_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield fp
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def _backup_sqlite(source_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{source_path.name}.backup_{ts}"
    src = sqlite3.connect(str(source_path), timeout=10.0)
    try:
        dst = sqlite3.connect(str(backup_path), timeout=10.0)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return backup_path


def main() -> int:
    args = _parse_args()
    trades_db = Path(args.trades_db)
    orders_db = Path(args.orders_db)
    until_dt = _parse_iso8601_utc(args.until_utc)

    if not trades_db.exists():
        print(f"[backfill] missing trades.db: {trades_db}")
        return 2
    if not orders_db.exists():
        print(f"[backfill] missing orders.db: {orders_db}")
        return 2

    lock_path = trades_db.with_suffix(trades_db.suffix + ".lock")
    backup_dir = Path(args.backup_dir)

    with _file_lock(lock_path):
        if not args.dry_run and not args.no_backup:
            backup_path = _backup_sqlite(trades_db, backup_dir)
            print(f"[backfill] backup created: {backup_path}")

        tcon = sqlite3.connect(str(trades_db), timeout=20.0)
        tcon.row_factory = sqlite3.Row
        ocon = sqlite3.connect(str(orders_db), timeout=10.0)
        ocon.row_factory = sqlite3.Row
        try:
            where = [
                "pocket IS NOT NULL",
                "LOWER(pocket) NOT IN ('manual', 'unknown')",
                "close_time IS NOT NULL",
                "("
                " entry_thesis IS NULL"
                " OR entry_thesis NOT LIKE '%entry_probability%'"
                " OR entry_thesis NOT LIKE '%entry_units_intent%'"
                ")",
            ]
            params: list[Any] = []
            if until_dt is not None:
                where.append("close_time <= ?")
                params.append(until_dt.isoformat())

            limit_sql = ""
            if int(args.limit or 0) > 0:
                limit_sql = "LIMIT ?"
                params.append(int(args.limit))

            rows = tcon.execute(
                f"""
                SELECT id, pocket, units, client_order_id, strategy_tag, entry_thesis, close_time
                FROM trades
                WHERE {' AND '.join(where)}
                ORDER BY id ASC
                {limit_sql}
                """,
                tuple(params),
            ).fetchall()

            print(f"[backfill] candidates={len(rows)} dry_run={bool(args.dry_run)} until_utc={args.until_utc or '-'}")
            if not rows:
                return 0

            updates: list[tuple[str, int]] = []
            recovered = 0
            fallback_only = 0
            sample_printed = 0

            for row in rows:
                trade_id = int(row["id"])
                client_id = str(row["client_order_id"] or "").strip()
                trade_units = row["units"]
                strategy_tag = str(row["strategy_tag"] or "").strip()

                existing = _safe_json_obj(row["entry_thesis"]) or {}

                recovered_thesis: dict[str, Any] | None = None
                if client_id:
                    attempt = ocon.execute(
                        """
                        SELECT request_json
                        FROM orders
                        WHERE client_order_id = ? AND status = 'submit_attempt'
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                        (client_id,),
                    ).fetchone()
                    if attempt and attempt["request_json"]:
                        recovered_thesis = _extract_thesis_from_request(attempt["request_json"])

                if isinstance(recovered_thesis, dict):
                    recovered += 1
                    merged = dict(existing)
                    merged.update(recovered_thesis)
                else:
                    fallback_only += 1
                    merged = dict(existing)

                if strategy_tag:
                    merged.setdefault("strategy_tag", strategy_tag)

                normalized_prob = _normalize_entry_probability(merged.get("entry_probability"))
                merged["entry_probability"] = normalized_prob if normalized_prob is not None else 1.0

                normalized_units = _normalize_units_intent(merged.get("entry_units_intent"))
                if normalized_units is None and trade_units is not None:
                    normalized_units = _normalize_units_intent(trade_units)
                merged["entry_units_intent"] = int(normalized_units or 0)

                new_json = json.dumps(merged, ensure_ascii=False, sort_keys=True)
                old_json = json.dumps(existing, ensure_ascii=False, sort_keys=True)
                if new_json == old_json:
                    continue
                updates.append((new_json, trade_id))

                if sample_printed < int(args.print_samples or 0):
                    print(
                        "[sample] id=%s pocket=%s client=%s close_time=%s old=%s new=%s"
                        % (
                            trade_id,
                            row["pocket"],
                            client_id or "-",
                            row["close_time"] or "-",
                            json.dumps(existing, ensure_ascii=False, sort_keys=True)[:160],
                            new_json[:160],
                        )
                    )
                    sample_printed += 1

            print(
                "[backfill] updates=%d recovered_from_orders=%d fallback_only=%d"
                % (len(updates), recovered, fallback_only)
            )

            if args.dry_run or not updates:
                return 0

            tcon.executemany(
                "UPDATE trades SET entry_thesis = ? WHERE id = ?",
                updates,
            )
            tcon.commit()
            print(f"[backfill] updated_rows={tcon.total_changes}")
            return 0
        finally:
            try:
                ocon.close()
            except Exception:
                pass
            try:
                tcon.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
