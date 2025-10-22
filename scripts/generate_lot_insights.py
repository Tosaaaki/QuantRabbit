#!/usr/bin/env python3
"""
BigQuery 上のトレード履歴からロット調整インサイトを生成し、
BigQuery テーブル / GCS に出力するユーティリティ。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analytics.lot_pattern_analyzer import LotPatternAnalyzer  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--trades-table", type=str, default=None)
    parser.add_argument("--insights-table", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--no-gcs", action="store_true", help="GCS 出力を無効化する")
    parser.add_argument(
        "--gcs-object",
        type=str,
        default="analytics/lot_insights.json",
        help="GCS へ書き出すパス（--no-gcs 指定時は無視）",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="生成結果を標準出力へ JSON として表示する",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    analyzer = LotPatternAnalyzer(
        project_id=args.project,
        dataset_id=args.dataset,
        trades_table=args.trades_table,
        insights_table=args.insights_table,
        lookback_days=args.lookback_days,
        min_trades=args.min_trades,
        gcs_bucket=None if args.no_gcs else None,
        gcs_object=args.gcs_object,
    )

    insights = analyzer.run()
    if args.print_json and insights:
        print(json.dumps([ins.__dict__ for ins in insights], ensure_ascii=False, indent=2))

    if not insights:
        logging.info("No insights generated.")
    else:
        logging.info("Generated %d insights.", len(insights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
