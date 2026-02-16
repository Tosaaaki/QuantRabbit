# SL Policy Map

SL（損切り）は「どこで決まるか」が複数レイヤに分かれています。  
このドキュメントは、USD/JPY 本番運用での優先順位と確認ポイントを整理したものです。

## 1. 生成順序（実運用の実体）
1. Worker が `entry_thesis.sl_pips` / `hard_stop_pips` / `tp_pips` を生成
2. `execution/order_manager.py` でエントリー前に再計算
   - dynamic SL (`ORDER_DYNAMIC_SL_*`)
   - min RR 調整 (`ORDER_MIN_RR_*`)
   - TP cap (`ORDER_TP_CAP_*`)
   - entry hard SL (`ORDER_ENTRY_HARD_STOP_PIPS*`)
3. OANDA 送信時の `stopLossOnFill` は `ORDER_FIXED_SL_MODE` だけで判定
4. fill 後の保護更新（`on_fill_protection`）
5. それでも SL が無い建玉は `order_manager.py` 側での保護更新処理が優先

## 1.5 固定SLの最短切替（唯一のON/OFF）
- `ORDER_FIXED_SL_MODE=1` で **全 pocket 共通で broker SL を付与**。
- `ORDER_FIXED_SL_MODE=0` で **全 pocket 共通で broker SL を停止**。
- 未設定は既定 OFF（明示的に `1`/`0` を置くのを推奨）

## 1.6 旧SL設定
- `ORDER_DISABLE_STOP_LOSS_*`、`ORDER_ENABLE_STOP_LOSS_*`、`ORDER_ALLOW_STOP_LOSS_WITH_EXIT_NO_NEGATIVE_CLOSE_*`
  は運用対象外（削除済み）。`ORDER_FIXED_SL_MODE` のみ実運用で管理する。

## 2. 重要な分岐（見落としやすい点）
- `ORDER_ENTRY_HARD_STOP_PIPS*` を設定しても、entry SL が無効の pocket では
  `stopLossOnFill` は付与されない（virtual SL としてのみ扱われる）。
- `quant-hard-stop-backfill.service` は現在の運用構成から除外。  
  SL 無し建玉の後付け処理はワーカー依存の緊急運用仕様ではなく、`order_manager.py` の通常フローを起点に再確認する。

## 2.5 SL関連キー（現行実装での役割）
- `ORDER_FIXED_SL_MODE`: Broker 送信SLのオン/オフ（運用時に本体として使う唯一のキー）
- `ORDER_ENTRY_HARD_STOP_PIPS*`, `ORDER_ENTRY_MAX_SL_PIPS*`: エントリー側で使う hard stop 距離の上限/基準（仮想SL系）
- `ORDER_ENTRY_LOSS_CAP_*`, `ORDER_ENTRY_LOSS_CAP_BUFFER_PIPS`: 1約定あたりの損失上限でサイズを間接制御
- `ORDER_DYNAMIC_SL_*`: `trade`/`tick`情報から virtual SLを再計算するための入力
- `ORDER_MIN_RR_*`: エントリーサイズに対する最小RR監査の閾値
- `ORDER_TP_CAP_*`: TP上限（間接的にSL/TPの実効範囲に影響）
- `ORDER_ROLLOVER_SL_STRIP_*`: ロールオーバー時間帯で既存SLの後処理を行う機能
- `ORDER_DISABLE_ENTRY_HARD_STOP_TAGS`: 戦略単位で hard stop 無効化（実データ送信の ON/OFF ではない）

## 3. 現行の推奨（精度優先 + 通常SL）
- `ops/env/entry_precision_hardening.env` を適用
  - 鮮度/密度ゲート強化（factor stale と microstructure）
  - `ORDER_FIXED_SL_MODE=1`
- 反映コマンド:
  - `scripts/vm_apply_entry_precision_hardening.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t`

## 4. 反映後の確認SQL（VM）
`orders.db` で `submit_attempt` の OANDA payload に `stopLossOnFill` が入っているか確認:

```sql
SELECT ts,pocket,status,
       json_extract(request_json,'$.oanda.order.stopLossOnFill.price') AS sl_on_fill,
       json_extract(request_json,'$.oanda.order.takeProfitOnFill.price') AS tp_on_fill
FROM orders
WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S','now','-2 hours')
  AND status='submit_attempt'
  AND pocket IN ('micro','scalp','macro')
ORDER BY ts DESC
LIMIT 30;
```

## 5. 障害時の切り分け
- SL が付かない:
  - `ops/env/quant-v2-runtime.env` で `ORDER_FIXED_SL_MODE` を確認（優先適用）
  - `systemctl show -p Environment quant-order-manager.service`
  - `orders.db` の `submit_attempt` payload に `stopLossOnFill` があるか
- SL が広すぎる:
  - `orders.db` での実送信 `stopLossOnFill` 価格と `TP` の乖離
  - `entry_precision_hardening` 前提の `ORDER_ENTRY_HARD_STOP_PIPS*` / `ORDER_DYNAMIC_SL_*` 設定
