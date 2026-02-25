# Observability & Logs

## 1. データ鮮度
- `max_data_lag_ms = 3000` 超は `DataFetcher` が `stale=True` を返し Risk Guard が発注拒否。
- Candle 確定は `tick.ts_ms // 60000` 変化で判定し、終値は最後の mid。
- `volume=0` は `missing_bar` としてログ。
- UI snapshot（`apps/autotune_ui.py`）は `generated_at` の鮮度を評価し、
  `UI_SNAPSHOT_MAX_AGE_SEC`（default: `max(120, UI_AUTO_REFRESH_SEC*4)`）を超えた stale 候補は
  `remote/gcs/local` の優先採用対象から除外する。全候補 stale の場合のみ最新時刻を採用する。
- `quant-ui-snapshot.service` 実行は `scripts/run_ui_snapshot.sh` で
  `UI_SNAPSHOT_MAX_RUNTIME_SEC`（default: `90`）の上限を持つ。
  上限超過時は `"[ui-snapshot] timed out ..."` を出して当該周期を打ち切り、
  次の timer 周期で自動再試行する。

## 2. ログの真偽と参照ルール
- 本番ログは VM `/home/tossaki/QuantRabbit/logs/` のみを真とする。
- ローカル `logs/*.db` は参考扱い。
- 運用上の指摘・報告・判断は必ず VM（ログ/DB/プロセス）または OANDA API を確認して行う。

## 3. ログ永続化とバックアップ
- GCS 自動退避: `GCS_BACKUP_BUCKET` を `ops/env/quant-v2-runtime.env` に設定し、`/etc/cron.hourly/qr-gcs-backup-core` で毎時アップロード。
- 保存先: `gs://$GCS_BACKUP_BUCKET/qr-logs/<hostname>/core_*.tar`

### Storage から読むとき（VM 上）
- 一覧: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core list`
- 最新を展開: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core latest /tmp/qr_gcs_restore`
- 指定取得: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core get gs://<bucket>/qr-logs/<host>/core_YYYYmmddTHHMMSSZ.tar /tmp/qr_gcs_restore`
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
