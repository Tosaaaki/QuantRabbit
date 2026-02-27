# Observability & Logs

## 1. データ鮮度
- `max_data_lag_ms = 3000` 超は `DataFetcher` が `stale=True` を返し Risk Guard が発注拒否。
- Candle 確定は `tick.ts_ms // 60000` 変化で判定し、終値は最後の mid。
- `volume=0` は `missing_bar` としてログ。
- UI snapshot（`apps/autotune_ui.py`）は `generated_at` の鮮度を評価し、
  `UI_SNAPSHOT_MAX_AGE_SEC`（default: `max(120, UI_AUTO_REFRESH_SEC*4)`）を超えた stale 候補は
  `remote/gcs/local` の優先採用対象から除外する。全候補 stale の場合のみ最新時刻を採用する。
- dashboard フロント（`templates/autotune/base.html`）の自動更新は
  timer/interval を単一インスタンスで管理し、`visibilitychange`・再スケジュール時に
  既存タイマーを必ず解放する。`data-auto-refresh` が無い画面では timer を保持しない。
- `hourly_trades` の fallback（`apps/autotune_ui.py`）は
  `snapshot.recent_trades` の件数上限ではなく `trades.db` の close 履歴を
  lookback 窓で再集計する。走査上限は
  `UI_HOURLY_FALLBACK_SCAN_LIMIT`（default: `5000`）。
  snapshot 側の `hourly_trades` は lookback/行数が不足する場合に採用せず、
  fallback 再集計へ切り替える。fallback の主経路は
  `julianday(close_time)` を使った時間窓 SQL 集計で、
  `UI_HOURLY_FALLBACK_QUERY_TIMEOUT_SEC`（default: `1.5`）の timeout を用いる。
- `scripts/publish_ui_snapshot.py` は `hourly_trades` を DB 走査で生成し、
  `UI_HOURLY_DB_TIMEOUT_SEC`（default: `1.5`）と
  `UI_HOURLY_DB_RETRY_COUNT`（default: `3`）で再試行する。
  DB集計が失敗した周期でも `recent_trades` から hourly payload を補完して
  `metrics.hourly_trades` の欠落を回避する。
- `scripts/run_sync_pipeline.py` も同じ hourly 生成ロジックを使って
  GCS `realtime/ui_state.json` を更新する。`--ui-recent` 既定は
  `UI_RECENT_TRADES_LIMIT`（default: `200`）に揃え、別経路更新でも
  history タブの集計窓を破壊しない。
- `quant-autotune-ui.service` は UI 応答を優先し、
  `POSITION_MANAGER_SERVICE_*TIMEOUT` を短く（1.2〜2.0s）設定して
  fail-fast 運用する。`POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=0` により
  heavy fallback を避け、軽量 snapshot へ降格する。
- `apps/autotune_ui.py` の dashboard 取得は、`remote/gcs` が fresh かつ
  metrics 有効なら `local` snapshot 構築を省略する。
  `fetch_attempts` には `local: skip (remote/gcs snapshot is already fresh)` を記録し、
  遅延が出る場合は `local` が本当に呼ばれているかをまず確認する。
- snapshot 採用は `remote > gcs > local` の固定順ではなく、fresh 候補の
  `hourly_trades` 有効性と metrics 充足数を優先して選択する。
  fresh な remote が軽量（hourly欠損）で gcs が完全な場合は gcs を優先する。
- `apps/autotune_ui.py` は secret を `UI_SECRET_CACHE_TTL_SEC`（default: 60）秒で
  TTLキャッシュし、未設定キーも負キャッシュする。
  Secret Manager 往復が遅い環境では、TTL切れ時のみ再取得が発生する。
- `apps/autotune_ui.py` は strategy control 状態を
  `UI_STRATEGY_CONTROL_CACHE_TTL_SEC`（default: 30）秒でTTLキャッシュする。
  `trades/signals/orders` 走査はTTL切れ時のみ実行されるため、
  dashboard の再描画遅延を抑えられる。
  `/ops/strategy-control` / `/api/strategy-control` の更新成功時は
  キャッシュを即時無効化し、次回描画で最新状態を再読み込みする。
  さらに `strategy_control.db` と `trades/signals/orders` の mtime を
  シグネチャとして比較し、TTL内でも外部更新を検知したら再計算する。
- `quant-ui-snapshot.service` 実行は `scripts/run_ui_snapshot.sh` で
  `UI_SNAPSHOT_MAX_RUNTIME_SEC`（default: `90`）の上限を持つ。
  上限超過時は `"[ui-snapshot] timed out ..."` を出して当該周期を打ち切り、
  次の timer 周期で自動再試行する。
- `scripts/publish_ui_snapshot.py --lite` は、既定では `sync_trades` を実行しない
  （`UI_SNAPSHOT_SYNC_TRADES_LITE=0` 既定）。`sync_trades` を有効化する場合のみ
  `UI_SNAPSHOT_SYNC_TRADES=1` と `UI_SNAPSHOT_SYNC_TRADES_LITE=1` を同時指定する。
- `scripts/publish_ui_snapshot.py --lite` は、既定では open positions 取得も行わない
  （`UI_SNAPSHOT_LITE_INCLUDE_POSITIONS=0` 既定）。建玉詳細を含める場合のみ
  `UI_SNAPSHOT_LITE_INCLUDE_POSITIONS=1` を指定する。
- `scripts/publish_ui_snapshot.py --lite` は、既定では拡張メトリクス
  （`orders_last` / `orders_status_1h` / `signals_recent` など）も収集しない
  （`UI_SNAPSHOT_LITE_INCLUDE_EXTENDED_METRICS=0` 既定）。
  拡張項目が必要な場合のみ `UI_SNAPSHOT_LITE_INCLUDE_EXTENDED_METRICS=1` を指定する。
- `scripts/publish_ui_snapshot.py --lite` は、既定では `PositionManager` を初期化しない
  （`UI_SNAPSHOT_LITE_USE_POSITION_MANAGER=0` 既定）。
  OANDA/position service 連携が必要な場合のみ
  `UI_SNAPSHOT_LITE_USE_POSITION_MANAGER=1` を指定する。
- GCS publish は `analytics/gcs_publisher.py` で
  `UI_SNAPSHOT_GCS_UPLOAD_TIMEOUT_SEC`（default: `8` 秒）の timeout を持つ。
  timeout 超過時は warning を出して、CLI/metadata 経路へフォールバックを試行する。

## 2. ログの真偽と参照ルール
- 本番ログは VM `/home/tossaki/QuantRabbit/logs/` のみを真とする。
- ローカル `logs/*.db` は参考扱い。
- 運用上の指摘・報告・判断は必ず VM（ログ/DB/プロセス）または OANDA API を確認して行う。

## 3. ログ永続化とバックアップ
- GCS 自動退避（guarded）:
  - `GCS_BACKUP_BUCKET` を `ops/env/quant-v2-runtime.env` に設定し、
    `quant-core-backup.timer` が `/usr/local/bin/qr-gcs-backup-core` を毎時実行する。
  - 既定で low-priority（`Nice=19`, `IOSchedulingClass=idle`）かつ
    `ops/env/quant-core-backup.env` の `load/D-state/mem/swap` ガードを満たさない場合は
    自動 `skip` する（トレード導線優先）。
  - SQLite は live `*.db-wal` を直接 tar せず、`.backup` で一時スナップショット化してから退避する。
  - legacy `/etc/cron.hourly/qr-gcs-backup-core` は `scripts/install_core_backup_service.sh` で無効化する。
- 保存先: `gs://$GCS_BACKUP_BUCKET/qr-logs/<hostname>/core_*.tar.gz`
- 2026-02-25 UTC 事象メモ:
  - `tar` が `orders.db-wal` を長時間保持すると `wal_checkpoint(TRUNCATE)` が進まず、
    `orders.db` stale と `database is locked` を誘発する。
  - `orders.db-wal` が異常肥大（数GB）した場合は、まず `lsof` で保持プロセスを特定して停止後、
    checkpoint を実行する。
  - 監査コマンド:
    - `sudo lsof /home/tossaki/QuantRabbit/logs/orders.db*`
    - `sqlite3 /home/tossaki/QuantRabbit/logs/orders.db 'PRAGMA wal_checkpoint(PASSIVE);'`
    - `sqlite3 /home/tossaki/QuantRabbit/logs/orders.db 'PRAGMA wal_checkpoint(TRUNCATE);'`

### Storage から読むとき（VM 上）
- 一覧: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core list`
- 最新を展開: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core latest /tmp/qr_gcs_restore`
- 指定取得: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core get gs://<bucket>/qr-logs/<host>/core_YYYYmmddTHHMMSSZ.tar.gz /tmp/qr_gcs_restore`
- 参照時の補助: `QR_BACKUP_HOST=<hostname>` を指定すると別ホストの退避も読める。

## 4. 例: 日次集計 / 直近オーダー

### 日次集計
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm sql \
  -f /home/tossaki/QuantRabbit/logs/trades.db \
  -q "SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;" -t
```

### 直近オーダー
```bash
gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap \
  --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
  --command "sqlite3 /home/tossaki/QuantRabbit/logs/orders.db 'select ts,pocket,side,units,client_order_id,status from orders order by ts desc limit 5;'"
```

## 5. 検証パイプライン
- `logs/replay/*.jsonl` で Record。
- Strategy Plugin は Backtest で再現性確認。
- Shadow では本番 tick + 仮想アカウントで `OrderIntent` と `risk_guard` 拒否理由を比較。
- 標準リプレイ手順は `docs/REPLAY_STANDARD.md` を参照。
- WFO / 過学習診断は `scripts/run_wfo_overfit_report.sh` を利用し、
  `logs/reports/wfo_overfit/latest.{json,md}` を監視する。

## 6. 観測指標と SLO
- 観測指標: `decision_latency_ms`, `data_lag_ms`, `order_success_rate`, `reject_rate`, `pnl_day_pips`, `drawdown_pct`。
- SLO: `decision_latency_ms p95 < 2000`, `order_success_rate ≥ 0.995`, `data_lag_ms p95 < 1500`, `drawdown_pct max < 0.18`。
- Alert: SLO 違反、`healthbeat` 欠損 5min 超、`order reject` 連続 3 件。
- V2 実行系では `decision_latency_ms` / `data_lag_ms` は
  `quant-strategy-control`（`workers/strategy_control/worker.py`）が市場オープン時に定期発行する。
  発行間隔は `STRATEGY_CONTROL_SLO_METRICS_INTERVAL_SEC`（既定 10 秒）で調整する。
- `scripts/policy_guard.py` は市場オープン時に SLO メトリクス欠損/停滞も違反扱いにする。
  - `POLICY_GUARD_REQUIRE_SLO_METRICS_WHEN_OPEN=1`（既定）
  - `POLICY_GUARD_SLO_METRICS_MAX_STALE_SEC=900`（既定）

## 7. metrics.db lock 運用ガード（2026-02-20）
- `utils/metrics_logger.py` は lock 競合時に指数バックオフ付きで再試行する。
- 調整キー（`ops/env/quant-v2-runtime.env`）:
  - `METRICS_DB_BUSY_TIMEOUT_MS`
  - `METRICS_DB_WRITE_RETRIES`
  - `METRICS_DB_RETRY_BASE_SLEEP_SEC`
  - `METRICS_DB_RETRY_MAX_SLEEP_SEC`
- `scripts/publish_range_mode.py` は DB 反映失敗時に
  `metric_write_failed metric=range_mode_active` を warning 出力する。
- `logged active=...` があるのに `range_mode_active` が更新されない場合は、
  まず上記 warning と `metrics_logger` の debug drop ログを確認する。
- `scripts/publish_range_mode.py` の鮮度判定は
  `M1` と `macro(H1/H4)` を別閾値で評価する。
  - `RANGE_MODE_PUBLISH_MAX_DATA_AGE_SEC`（M1 側、既定 900 秒）
  - `RANGE_MODE_PUBLISH_MACRO_MAX_DATA_AGE_SEC`（macro 側、未設定時は TF から自動算出）
  - `range_mode_active` の tags で `m1_age_sec` / `macro_age_sec` /
    `m1_limit_sec` / `macro_limit_sec` を確認する。

## 8. マージン余力判定の落とし穴（2025-12-29 対応済み）
- fast_scalp が shared_state 欠落時に余力0と誤判定し全シグナルをスキップした事象あり。
- `workers/fast_scalp/worker.py` で `get_account_snapshot()` フォールバックを追加済み。
- 再発時はログに `refreshed account snapshot equity=... margin_available=...` が出ることを確認。
- 0 判定が続く場合は OANDA snapshot 取得を先に疑う。

## 9. Forecast 復旧監視（2026-02-21）
- `quant-forecast-watchdog.timer` が 1 分周期で `scripts/forecast_watchdog.sh` を実行する。
- 監視対象:
  - `http://127.0.0.1:8302/health`
  - `quant-forecast.service` の実応答可否（active 表示のみを信用しない）
- 振る舞い:
  - `FORECAST_WATCHDOG_MAX_FAILS` 連続失敗で `quant-forecast.service` を再起動。
  - 再起動後も不健康なら `FORECAST_WATCHDOG_DISABLE_BQ_ON_ESCALATE=1` 時に
    `quant-bq-sync.service` を停止して forecast 優先で復旧。
  - `quant-bq-sync` 側は `BQ_FAILURE_BACKOFF_BASE_SEC` / `BQ_FAILURE_BACKOFF_MAX_SEC`
    で export 失敗時の cooldown を持ち、外部SSL失敗の連発時は自動的に再送を間引く。
- 監査コマンド:
  - `systemctl status quant-forecast-watchdog.timer quant-forecast-watchdog.service`
  - `journalctl -u quant-forecast-watchdog.service -n 100 --no-pager`
  - `curl -sS http://127.0.0.1:8302/health`
