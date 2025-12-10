#!/usr/bin/env python3
"""VM 上でトレード同期と BigQuery 送信を連続実行するパイプライン。"""

from __future__ import annotations

import argparse
import logging
import shutil
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
from analytics.lot_pattern_analyzer import LotPatternAnalyzer
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
        metrics = pm.get_performance_summary()
        try:
            gcs_publisher.publish_snapshot(
                new_trades=new_trades,
                recent_trades=recent_trades,
                open_positions=open_positions,
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("[PIPELINE] GCS スナップショット更新に失敗: %s", exc)

    return new_trades


def _sync_remote_logs(dest_dir: Path) -> None:
    """Copy key SQLite/log snapshots to a remote_logs directory."""

    target = dest_dir.expanduser()
    target.mkdir(parents=True, exist_ok=True)
    files = [
        Path("logs/trades.db"),
        Path("logs/orders.db"),
        Path("logs/metrics.db"),
        Path("logs/oanda/candles_M1_latest.json"),
        Path("logs/oanda/candles_H1_latest.json"),
        Path("logs/oanda/candles_H4_latest.json"),
    ]
    copied = 0
    for src in files:
        if src.exists():
            shutil.copy2(src, target / src.name)
            copied += 1
    oanda_dir = Path("logs/oanda")
    if oanda_dir.exists():
        txns = sorted(oanda_dir.glob("transactions_*.jsonl"))
        if txns:
            latest = txns[-1]
            shutil.copy2(latest, target / latest.name)
            copied += 1
            meta = latest.with_name(f"{latest.stem}_meta.json")
            if meta.exists():
                shutil.copy2(meta, target / meta.name)
                copied += 1
    logging.info("[REMOTE] synced %d files to %s", copied, target)


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
    parser.add_argument(
        "--disable-lot-insights",
        action="store_true",
        help="BigQuery 解析によるロットインサイト生成を無効化する。",
    )
    parser.add_argument(
        "--disable-bq",
        action="store_true",
        help="BigQuery エクスポートを完全に無効化する。Lot insights もスキップ。",
    )
    parser.add_argument(
        "--remote-dir",
        type=Path,
        default=None,
        help="Sync trades/candles snapshots to the given directory (e.g. remote_logs_current).",
    )
    parser.add_argument(
        "--remote-sync-interval",
        type=float,
        default=300.0,
        help="Minimum seconds between remote log syncs (default: 300).",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)
    remote_dir = args.remote_dir.expanduser() if args.remote_dir else None

    exporter = None
    if not args.disable_bq:
        try:
            exporter = BigQueryExporter()
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(
                "[PIPELINE] BigQuery exporter unavailable (%s). disable-bq fallback.",
                exc,
            )
            args.disable_bq = True
    gcs_publisher = None if args.disable_gcs else GCSRealtimePublisher()
    analyzer = None
    if (not args.disable_bq) and (not args.disable_lot_insights) and exporter:
        try:
            analyzer = LotPatternAnalyzer()
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("[PIPELINE] Lot insights disabled (%s).", exc)
            analyzer = None
    pm = PositionManager()
    stop_requested = False
    last_bq_export = 0.0
    bq_interval = args.bq_interval
    bq_on_new = not args.no_bq_on_new
    last_remote_sync = 0.0

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
                if remote_dir and args.remote_sync_interval > 0:
                    if now - last_remote_sync >= args.remote_sync_interval:
                        try:
                            _sync_remote_logs(remote_dir)
                            last_remote_sync = now
                        except Exception as exc:  # noqa: BLE001
                            logging.exception("[REMOTE] sync failed: %s", exc)
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

                if run_bq and exporter is None:
                    run_bq = False
                if run_bq and exporter:
                    stats = exporter.export(limit=args.limit or _BQ_MAX_EXPORT)
                    logging.info(
                        "[PIPELINE] export done rows=%s last_updated=%s",
                        stats.exported,
                        stats.last_updated_at,
                    )
                    if analyzer:
                        try:
                            insights = analyzer.run()
                            logging.info(
                                "[PIPELINE] lot insights generated=%d",
                                len(insights),
                            )
                        except Exception as exc:  # noqa: BLE001
                            logging.exception("[PIPELINE] lot insights 生成に失敗: %s", exc)
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
