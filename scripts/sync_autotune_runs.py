#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Push existing SQLite autotune runs into BigQuery."""
import argparse
from pathlib import Path

from autotune.database import (
    DEFAULT_DB_PATH,
    record_run_bigquery,
    get_connection,
    list_runs,
    dump_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local autotune runs to BigQuery")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite DB")
    parser.add_argument("--limit", type=int, default=500, help="Maximum rows to sync")
    args = parser.parse_args()

    conn = get_connection(Path(args.db))
    rows = list_runs(conn, status=None, limit=args.limit)
    print(f"[INFO] exporting {len(rows)} rows to BigQuery")
    for row in rows:
        rec = dump_dict(row)
        record_run_bigquery(
            run_id=rec["run_id"],
            strategy=rec["strategy"],
            params=rec.get("params"),
            train=rec.get("train"),
            valid=rec.get("valid"),
            score=rec.get("score", 0.0),
            source_file=rec.get("source_file"),
        )
    print("[INFO] sync completed")


if __name__ == "__main__":
    main()
