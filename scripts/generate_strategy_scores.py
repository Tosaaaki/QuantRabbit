#!/usr/bin/env python3
"""
BigQuery trades_raw を集計し、戦略ごとのスコア/ロット係数/SLTP 推奨値を
Firestore にサマリ書き込みするユーティリティ。

デフォルトでは Firestore に書き込み、--dry-run なら標準出力だけ。
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

from analytics.strategy_score_exporter import (  # noqa: E402
    StrategyScoreExporter,
    payload_summary,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--table", type=str, default=None)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument(
        "--collection",
        type=str,
        default="strategy_scores",
        help="Firestore collection name",
    )
    parser.add_argument(
        "--document",
        type=str,
        default="current",
        help="Firestore document name",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write payload JSON (in addition to Firestore)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Firestore")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    exporter = StrategyScoreExporter(
        project_id=args.project,
        dataset_id=args.dataset or exporter_default("dataset"),
        trades_table=args.table or exporter_default("table"),
        lookback_days=args.lookback_days,
        min_trades=args.min_trades,
        firestore_collection=args.collection,
        firestore_document=args.document,
    )
    payload = exporter.build_payload()
    logging.info("[SUMMARY] %s", payload_summary(payload))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logging.info("[WRITE] payload json -> %s", args.json_out)

    if args.dry_run:
        logging.info("[DRY RUN] skip Firestore write")
        return 0
    try:
        exporter.push_firestore(payload)
    except Exception as exc:  # noqa: BLE001
        logging.error("[ERROR] Firestore write failed: %s", exc)
        return 1
    return 0


def exporter_default(kind: str) -> str | None:
    """Small helper so argparse defaults stay minimal."""
    from analytics import strategy_score_exporter as sse

    if kind == "dataset":
        return sse._DEFAULT_DATASET  # type: ignore[attr-defined]
    if kind == "table":
        return sse._DEFAULT_TABLE  # type: ignore[attr-defined]
    return None


if __name__ == "__main__":
    raise SystemExit(main())

