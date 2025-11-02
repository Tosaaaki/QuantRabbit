#!/usr/bin/env python3
"""CLI helper to export recent trades into BigQuery."""

from __future__ import annotations

import argparse
import logging

from analytics.bq_exporter import BigQueryExporter, _DB_DEFAULT, _STATE_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5000, help="Max rows to export in one batch")
    parser.add_argument("--sqlite", default=None, help="Override path to trades.db")
    parser.add_argument("--state", default=None, help="Override path to sync state JSON")
    args = parser.parse_args()

    exporter = BigQueryExporter(
        sqlite_path=args.sqlite or _DB_DEFAULT,
        state_path=args.state or _STATE_DEFAULT,
    )
    stats = exporter.export(limit=args.limit)
    logging.info("Exported %s rows (up to %s)", stats.exported, stats.last_updated_at)


if __name__ == "__main__":
    main()
