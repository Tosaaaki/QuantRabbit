# Trade Findings – 2025-11-10

## Data snapshot
- Trades: remote_logs_vm/trades_vm.db (synced 2025-11-10T21:46:03Z)
- M1 candles: tmp/candles_M1_20251110.json (downloaded from OANDA v3 API)
- Macro snapshot: fixtures/macro_snapshots/latest.json (asof 2025-11-10T22:44:21Z)

## Observations
1. **Macro pocket停滞** – snapshot refresh間隔30分のままだと `stale>900s` が必ず発生し、24時間以上 `focus=hybrid` / `weight_macro<=0.18` に縮退。macro戦略は完全停止していた。
2. **微量トレード** – 25件のうち 22件がBB_RSIで、0:00–07:00 UTCに集中。ロンドン立ち上がり(07:00–12:00)とNY序盤は4件のみ。手動ショート40kが証拠金を専有して `allowed_lot()` が 0.05〜0.08lot に制限されている。
3. **チャンス逸失** – M1ローソクの30分ウィンドウで以下の未エントリが確認できる。
   | Window (UTC) | Move | Direction | Notes |
   |--------------|------|-----------|-------|
   | 07:16–07:35  | 15.2 pips | down | ロンドン早朝急落。microはcooldown、macro閉鎖でノートレード。
   | 10:32–10:55  | 11.8 pips | up   | NY先物オープンに被る反騰。スパイク/Trend系なし。
   | 13:05–13:30  | 9.4 pips  | down | 東京午後〜欧州序盤の調整。spread<1pだが micro冷却で休眠。

## 次のアクション
1. **Macro snapshot** – asofベースで10分毎に再生成。`main.py` で stale 検知時に即 `refresh_macro_snapshot()` を呼び出し、自動復帰できるようにする。
2. **Sync pipeline** – `scripts/run_sync_pipeline.py` に `--disable-bq` を追加し、権限が無い環境でも trades/candles 同期だけを実行できるようにする。
3. **戦略補填** – London momentum / VWAP 逆張りの追加と、macro/micro 両方での `spread-aware lot scaling` 調整を検討する。
