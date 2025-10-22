#!/usr/bin/env python3
"""VM 上でトレード同期と BigQuery 送信を連続実行するパイプライン。"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analytics.bq_exporter import BigQueryExporter, _BQ_MAX_EXPORT
from analytics.gcs_publisher import GCSRealtimePublisher
from execution.position_manager import PositionManager

LOG_FILE = Path("logs/pipeline.log")


def _setup_logging(verbose: bool) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _run_cycle(
    pm: PositionManager, gcs_publisher: GCSRealtimePublisher | None, ui_recent: int
) -> list[dict]:
    logging.info("[PIPELINE] sync_trades start")
    new_trades = pm.sync_trades()

    if gcs_publisher and gcs_publisher.enabled:
        try:
            open_positions = pm.get_open_positions()
        except Exception as exc:  # noqa: BLE001
            logging.exception("[PIPELINE] open positions 取得に失敗: %s", exc)
            open_positions = {}
        recent_trades = pm.fetch_recent_trades(limit=ui_recent)
        try:
            gcs_publisher.publish_snapshot(
                new_trades=new_trades,
                recent_trades=recent_trades,
                open_positions=open_positions,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("[PIPELINE] GCS スナップショット更新に失敗: %s", exc)

    return new_trades


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=120.0,
        help="ループ間隔（秒）。--once 指定時は無視される。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="1 回の BigQuery export 上限行数。未指定なら環境変数に従う。",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="1 サイクルのみ実行して終了。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG ログを有効化。",
    )
    parser.add_argument(
        "--bq-interval",
        type=float,
        default=600.0,
        help="BigQuery へのエクスポート実行間隔（秒）。0 以下で新規トレード発生時のみ。",
    )
    parser.add_argument(
        "--no-bq-on-new",
        action="store_true",
        help="新規トレードがあっても即時 BigQuery export を行わない。",
    )
    parser.add_argument(
        "--ui-recent",
        type=int,
        default=50,
        help="UI スナップショットに含める直近トレード件数。",
    )
    parser.add_argument(
        "--disable-gcs",
        action="store_true",
        help="GCS へのリアルタイム出力を抑止する。",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    exporter = BigQueryExporter()
    gcs_publisher = None if args.disable_gcs else GCSRealtimePublisher()
    pm = PositionManager()
    stop_requested = False
    last_bq_export = 0.0
    bq_interval = args.bq_interval
    bq_on_new = not args.no_bq_on_new

    def _handle_signal(signum: int, _frame: Optional[object]) -> None:
        nonlocal stop_requested
        logging.info("[PIPELINE] signal %s を受信。終了準備...", signum)
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        while True:
            try:
                new_trades = _run_cycle(pm, gcs_publisher, args.ui_recent)

                now = time.monotonic()
                run_bq = False
                if args.once:
                    run_bq = True
                elif bq_interval <= 0:
                    run_bq = bool(new_trades) if bq_on_new else False
                else:
                    if now - last_bq_export >= bq_interval:
                        run_bq = True
                    elif bq_on_new and new_trades and now - last_bq_export >= 5.0:
                        # 新規トレードがあり、最後の export から短時間経過した場合でも水平分散
                        run_bq = True

                if run_bq:
                    stats = exporter.export(limit=args.limit or _BQ_MAX_EXPORT)
                    logging.info(
                        "[PIPELINE] export done rows=%s last_updated=%s",
                        stats.exported,
                        stats.last_updated_at,
                    )
                    last_bq_export = now
            except Exception as exc:  # noqa: BLE001
                logging.exception("[PIPELINE] サイクル失敗: %s", exc)
            if args.once or stop_requested:
                break
            sleep_for = max(args.interval, 5.0)
            logging.debug("[PIPELINE] sleeping %.1fs", sleep_for)
            for _ in range(int(sleep_for)):
                if stop_requested:
                    break
                time.sleep(1)
            if stop_requested:
                break
            residual = sleep_for - int(sleep_for)
            if residual > 0 and not stop_requested:
                time.sleep(residual)
    finally:
        pm.close()

    logging.info("[PIPELINE] shutdown complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
