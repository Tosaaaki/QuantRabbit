#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from typing import Iterable

from google.cloud import bigquery


def _client(project: str | None) -> bigquery.Client:
    if project:
        return bigquery.Client(project=project)
    return bigquery.Client()


def _fetch_rows(
    client: bigquery.Client,
    dataset: str,
    table: str,
    lookbacks: Iterable[int],
) -> list[dict]:
    lookbacks = sorted({max(int(lb), 1) for lb in lookbacks})
    if not lookbacks:
        lookbacks = [7, 30]
    lookback_sql = "\n    UNION ALL\n    ".join(
        f"""SELECT {lb} AS lookback, pocket, strategy, close_time, pl_pips, realized_pl
        FROM `{client.project}.{dataset}.{table}`
        WHERE state = 'CLOSED' AND close_time IS NOT NULL
          AND close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lb} DAY)"""
        for lb in lookbacks
    )
    query = f"""
    WITH windows AS (
      {lookback_sql}
    )
    SELECT
      lookback,
      pocket,
      strategy,
      COUNT(*) AS trades,
      SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
      SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
      AVG(pl_pips) AS avg_pips,
      SAFE_DIVIDE(
        SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END),
        NULLIF(ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)), 0)
      ) AS pf,
      SUM(realized_pl) AS realized_pl
    FROM windows
    GROUP BY lookback, pocket, strategy
    ORDER BY lookback, pocket, strategy
    """
    rows = client.query(query).result()
    output = []
    for row in rows:
        output.append(
            {
                "lookback": int(row["lookback"]),
                "pocket": row["pocket"],
                "strategy": row["strategy"],
                "trades": int(row["trades"] or 0),
                "wins": int(row["wins"] or 0),
                "losses": int(row["losses"] or 0),
                "avg_pips": float(row["avg_pips"] or 0.0),
                "pf": None if row["pf"] is None or (isinstance(row["pf"], float) and not math.isfinite(row["pf"])) else float(row["pf"]),
                "realized_pl": float(row["realized_pl"] or 0.0),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BigQuery のトレード集計を出力（lookback 指定可）"
    )
    parser.add_argument("--project", default=None, help="BigQuery プロジェクト (既定: 環境変数)")
    parser.add_argument("--dataset", default="quantrabbit", help="BigQuery データセット ID")
    parser.add_argument("--table", default="trades_raw", help="トレードテーブル名")
    parser.add_argument(
        "--lookback",
        type=int,
        action="append",
        help="集計対象の日数 (複数指定可, 例: --lookback 7 --lookback 30)",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args()

    client = _client(args.project)
    rows = _fetch_rows(client, args.dataset, args.table, args.lookback or (7, 30))
    if args.format == "json":
        import json

        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    headers = ["lookback", "pocket", "strategy", "trades", "wins", "losses", "avg_pips", "pf", "realized_pl"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(f"{row.get(h, '')}"))
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for row in rows:
        values = []
        for h in headers:
            val = row.get(h)
            if isinstance(val, float):
                val = f"{val:.3f}"
            values.append(str(val).ljust(widths[h]))
        print(" | ".join(values))


if __name__ == "__main__":
    main()
